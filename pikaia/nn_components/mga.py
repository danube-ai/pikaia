from typing import Optional

import torch
import torch.nn as nn


class GeneticFitness(nn.Module):
    """
    Computes genetic fitness scores from masked attention weights matrix.

    This module implements genetic sorting on the attention matrix to modulate
    attention weights per token:
    1. Scale attention weights to [0, 1] range using sigmoid
    2. Apply genetic sorting to compute fitness scores per token
    3. Fitness scores weight the attention weights for each query token
    """

    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size

    def forward(self, attn_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute genetic fitness scores from attention weights.

        Args:
            attn_matrix: Attention weights matrix of shape (batch_size, num_heads, seq_length, seq_length)
                         Already masked with 0s in invalid positions (outside window/causal).

        Returns:
            gene_fitness: Fitness scores of shape (batch_size, num_heads, seq_length)
                         One fitness score per query token.
        """
        B, H, T, _ = attn_matrix.shape

        # 1. Scale attention weights to [0, 1] using sigmoid
        scaled_attn = torch.sigmoid(attn_matrix)  # (B, H, T, T)

        # 2. Apply genetic sorting on scaled attention matrix
        # scaled_attn: (B, H, T, T)
        gene_means = torch.mean(scaled_attn, dim=-1)  # (B, H, T) mean over j for each i
        denom = gene_means + 0.5  # (B, H, T)
        sum_inv_denom = torch.sum(
            1 / denom, dim=2, keepdim=True
        )  # (B, H, 1) sum over T
        gene_fitness = 1 / (denom * sum_inv_denom)  # (B, H, T)

        return gene_fitness


class MultiHeadGeneticAttention(nn.Module):
    """
    Multi-Head Genetic Attention module.

    Implements genetic attention following classical Multi-Head Attention (MHA) steps
    with genetic sorting applied to the attention weights matrix:

    1. Compute Q, K, V projections
    2. Apply QK RMS normalization
    3. Calculate attention weights: Q @ K^T / sqrt(d_k)
    4. Apply windowed causal masking
    5. Compute gene fitness scores from masked attention matrix
    6. Weight attention weights by fitness scores (element-wise per row)
    7. Apply softmax and aggregate with values

    Features:
    - Windowed causal attention (sliding window + causal masking)
    - QK RMS normalization (always applied)
    - Optional Multi-Head Latent Attention (MLA) for input compression
    - Grouped Query Attention (GQA) support
    - Gated DeltaNet gating mechanism

    The genetic component modulates attention weights based on evolutionary fitness
    computed from the attention matrix itself, rather than from token embeddings.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        num_value_heads: Optional[int] = None,
        use_mla: bool = False,
        mla_compression_dim: Optional[int] = None,
        use_gated_deltanet: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.use_mla = use_mla
        self.use_gated_deltanet = use_gated_deltanet
        self.dropout = nn.Dropout(dropout)

        # GQA: Allow fewer value heads than query/key heads
        self.num_value_heads = (
            num_value_heads if num_value_heads is not None else num_heads
        )
        if num_heads % self.num_value_heads != 0:
            raise ValueError("num_heads must be divisible by num_value_heads")
        self.num_value_groups = num_heads // self.num_value_heads

        # Genetic Fitness Module (computes fitness from attention matrix)
        self.genetic_fitness = GeneticFitness(window_size=window_size)

        # Projections
        if self.use_mla:
            # MLA: Low-rank compression for inputs
            if mla_compression_dim is None:
                mla_compression_dim = int(d_model * 0.5)  # Default compression

            self.mla_down = nn.Linear(d_model, mla_compression_dim)
            self.mla_up_q = nn.Linear(mla_compression_dim, d_model)
            self.mla_up_k = nn.Linear(mla_compression_dim, d_model)
            self.mla_up_v = nn.Linear(
                mla_compression_dim, self.num_value_heads * self.head_dim
            )

            # Layer Norms for MLA
            self.ln_q = nn.LayerNorm(d_model)
            self.ln_k = nn.LayerNorm(d_model)
            self.ln_v = nn.LayerNorm(self.num_value_heads * self.head_dim)

        else:
            # Standard MHA/GQA projections
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, self.num_value_heads * self.head_dim)

        # QK RMS Norms (always applied)
        self.rms_q = nn.RMSNorm(self.head_dim)
        self.rms_k = nn.RMSNorm(self.head_dim)

        self.out_proj = nn.Linear(d_model, d_model)

        # Gated DeltaNet: Gating mechanism
        if self.use_gated_deltanet:
            self.gate_proj = nn.Linear(d_model, d_model)
            self.conv1d = nn.Conv1d(
                d_model, d_model, kernel_size=3, padding=1, groups=d_model
            )
            self.act = nn.SiLU()

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass implementing genetic attention.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            attention_mask: Optional attention mask (currently unused, masking is built-in)

        Returns:
            Output tensor of shape (batch_size, seq_length, d_model)
        """
        B, T, D = x.shape

        # 1. Project inputs to Q, K, V
        if self.use_mla:
            # MLA: Compress inputs then project to Q, K, V
            compressed = self.mla_down(x)
            q = self.ln_q(self.mla_up_q(compressed))
            k = self.ln_k(self.mla_up_k(compressed))
            v = self.ln_v(self.mla_up_v(compressed))
        else:
            # Standard projections
            q = self.w_q(x)
            k = self.w_k(x)
            v = self.w_v(x)

        # 2. Reshape for multi-head attention
        # Q: (B, T, H, D_h) -> (B, H, T, D_h)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # K: (B, T, H, D_h) -> (B, H, T, D_h)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # V: (B, T, H_v, D_h) -> (B, H_v, T, D_h)
        v = v.view(B, T, self.num_value_heads, self.head_dim).transpose(1, 2)

        # 3. Apply QK RMS normalization
        q = self.rms_q(q)
        k = self.rms_k(k)

        # 4. Compute attention weights: Q @ K^T / sqrt(d_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # (B, H, T, T)

        # 5. Apply windowed causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=x.device))  # (T, T)
        window_mask = torch.zeros(T, T, device=x.device)
        for i in range(T):
            start_j = max(0, i - self.window_size)
            window_mask[i, start_j : i + 1] = 1

        combined_mask = causal_mask * window_mask  # (T, T)
        combined_mask = combined_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        # Mask attention: set invalid positions to 0
        masked_attn = attn * combined_mask

        # 6. Compute genetic fitness scores from masked attention matrix
        gene_fitness = self.genetic_fitness(masked_attn)  # (B, H, T)

        # 7. Weight attention weights by fitness scores (element-wise per row)
        weighted_attn = masked_attn * gene_fitness.unsqueeze(-1)  # (B, H, T, T)

        # 8. Apply softmax (mask invalid positions with -inf)
        softmax_input = weighted_attn + (1 - combined_mask) * (-1e10)
        attn_weights = torch.softmax(softmax_input, dim=-1)

        # 9. Apply attention to values
        # Handle GQA: expand V to match number of heads if needed
        if self.num_value_groups > 1:
            v = v.unsqueeze(2).expand(-1, -1, self.num_value_groups, -1, -1)
            v = v.reshape(B, self.num_heads, T, self.head_dim)

        # Weighted sum: (B, H, T, T) @ (B, H, T, D_h) -> (B, H, T, D_h)
        out = torch.matmul(attn_weights, v)

        # 10. Reshape and apply output projection
        # (B, H, T, D_h) -> (B, T, H, D_h) -> (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # 11. Apply Gated DeltaNet gating if enabled
        if self.use_gated_deltanet:
            # Convolution-based gating mechanism
            gate_input = x.transpose(1, 2)  # (B, D, T)
            gate = self.act(self.conv1d(gate_input)).transpose(1, 2)  # (B, T, D)
            gate = self.gate_proj(gate)  # (B, T, D)
            out = out * torch.sigmoid(gate)

        out = self.out_proj(out)

        return self.dropout(out)
