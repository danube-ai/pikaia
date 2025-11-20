import time
from typing import Optional

import torch
import torch.nn as nn

from pikaia.config.logger import logger


class GeneticFitness(nn.Module):
    """
    Computes genetic fitness scores from correlations within a sliding window.

    This module implements correlation-based genetic attention where:
    1. Correlations between token pairs within window are computed
    2. Genetic sorting is applied to the correlation matrix
    3. Gene fitness weights the correlations for attention
    """

    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: Shape (batch_size, num_heads, seq_length, head_dim)

        Returns:
            attention_weights: Shape (batch_size, num_heads, seq_length, seq_length)
        """
        B, H, T, D = token_embeddings.shape
        device = token_embeddings.device

        start_time = time.perf_counter()

        # 1. Compute all pairwise correlations using vectorized operations
        # emb: (B, H, T, D)
        emb_i = token_embeddings.unsqueeze(3)  # (B, H, T, 1, D)
        emb_j = token_embeddings.unsqueeze(2)  # (B, H, 1, T, D)

        # Compute dot products for all pairs
        dot = (emb_i * emb_j).sum(-1)  # (B, H, T, T)

        # Compute norms
        norm_i = torch.norm(token_embeddings, dim=-1).unsqueeze(-1)  # (B, H, T, 1)
        norm_j = torch.norm(token_embeddings, dim=-1).unsqueeze(2)  # (B, H, 1, T)

        # Compute correlations
        correlations = dot / (norm_i * norm_j + 1e-8)  # (B, H, T, T)
        correlations = torch.nan_to_num(correlations, nan=0.0, posinf=1.0, neginf=-1.0)

        # Mask out invalid positions (outside window and non-causal)
        # Create mask for valid positions
        causal_mask = torch.tril(torch.ones(T, T, device=device))  # (T, T)
        window_mask = torch.zeros(T, T, device=device)
        for i in range(T):
            start_j = max(0, i - self.window_size)
            window_mask[i, start_j : i + 1] = 1

        combined_mask = causal_mask * window_mask  # (T, T)
        combined_mask = combined_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        # Zero out invalid correlations
        correlations = correlations * combined_mask

        logger.info(
            f"GeneticFitness: Correlation computation took {time.perf_counter() - start_time:.4f}s"
        )
        start_time = time.perf_counter()

        # 2. Scale correlations to [0, 1] range
        # Correlations are in [-1, 1], shift to [0, 1]
        correlations = (correlations + 1) / 2  # (B, H, T, T)

        logger.info(
            f"GeneticFitness: Scaling correlations took {time.perf_counter() - start_time:.4f}s"
        )
        start_time = time.perf_counter()

        # 3. Apply genetic sorting on correlation matrix
        # Vectorized: compute gene fitness for all batches and heads at once
        # correlations: (B, H, T, T)
        gene_means = torch.mean(correlations, dim=3)  # (B, H, T) mean over j for each i
        denom = gene_means + 0.5  # (B, H, T)
        sum_inv_denom = torch.sum(
            1 / denom, dim=2, keepdim=True
        )  # (B, H, 1) sum over T
        gene_fitness = 1 / (denom * sum_inv_denom)  # (B, H, T)

        logger.info(
            f"GeneticFitness: Genetic sorting took {time.perf_counter() - start_time:.4f}s"
        )
        start_time = time.perf_counter()

        # 4. Compute weighted correlations
        # gene_fitness: (B, H, T) -> (B, H, T, 1)
        # correlations: (B, H, T, T)
        weighted_corr = correlations * gene_fitness.unsqueeze(-1)  # (B, H, T, T)

        logger.info(
            f"GeneticFitness: Weighted correlations took {time.perf_counter() - start_time:.4f}s"
        )
        start_time = time.perf_counter()

        # 5. Apply softmax by row to get attention weights
        # Mask out invalid positions (outside window and non-causal)
        causal_mask = torch.tril(torch.ones(T, T, device=device))  # (T, T)
        window_mask = torch.zeros(T, T, device=device)
        for i in range(T):
            start_j = max(0, i - self.window_size)
            window_mask[i, start_j : i + 1] = 1

        combined_mask = causal_mask * window_mask  # (T, T)
        combined_mask = combined_mask.view(1, 1, T, T)  # (1, 1, T, T)

        # Apply mask
        masked_corr = weighted_corr * combined_mask + (1 - combined_mask) * (-1e10)

        # Softmax
        attention_weights = torch.softmax(masked_corr, dim=-1)

        logger.info(
            f"GeneticFitness: Softmax took {time.perf_counter() - start_time:.4f}s"
        )

        return attention_weights


class MultiHeadGeneticAttention(nn.Module):
    """
    Multi-Head Genetic Attention module.

    Implements genetic attention with support for:
    - Multi-Head Attention (MHA)
    - Grouped Query Attention (GQA) concepts (fewer value heads)
    - Multi-Head Latent Attention (MLA) concepts (low-rank compression)
    - Sliding Window Attention (SWA)
    - Gated DeltaNet concepts (gating mechanism)
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
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.use_mla = use_mla
        self.use_gated_deltanet = use_gated_deltanet
        self.dropout = nn.Dropout(dropout)

        # GQA: Allow fewer value heads than genetic heads
        self.num_value_heads = (
            num_value_heads if num_value_heads is not None else num_heads
        )
        if num_heads % self.num_value_heads != 0:
            raise ValueError("num_heads must be divisible by num_value_heads")
        self.num_value_groups = num_heads // self.num_value_heads

        # Genetic Fitness Module (Attention Core)
        self.genetic_fitness = GeneticFitness(window_size=window_size)

        # Projections
        if self.use_mla:
            # MLA: Low-rank compression for inputs
            if mla_compression_dim is None:
                mla_compression_dim = int(d_model * 0.5)  # Default compression

            self.mla_down = nn.Linear(d_model, mla_compression_dim)
            self.mla_up_g = nn.Linear(mla_compression_dim, d_model)
            self.mla_up_v = nn.Linear(
                mla_compression_dim, self.num_value_heads * self.head_dim
            )

            # Layer Norms for MLA
            self.ln_g = nn.LayerNorm(d_model)
            self.ln_v = nn.LayerNorm(self.num_value_heads * self.head_dim)

        else:
            # Standard MHA/GQA projections
            self.w_g = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, self.num_value_heads * self.head_dim)

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
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            attention_mask: Optional mask (currently not used, causal masking is built-in)

        Returns:
            Output tensor of shape (batch_size, seq_length, d_model)
        """
        B, T, D = x.shape

        start_time = time.perf_counter()

        # 1. Projections to G (Genetic/Key) and V (Value)
        if self.use_mla:
            # MLA: Down-project then Up-project
            compressed = self.mla_down(x)
            g = self.ln_g(self.mla_up_g(compressed))
            v = self.ln_v(self.mla_up_v(compressed))
        else:
            # Standard
            g = self.w_g(x)
            v = self.w_v(x)

        logger.info(
            f"MultiHeadGeneticAttention: Projections took {time.perf_counter() - start_time:.4f}s"
        )
        start_time = time.perf_counter()

        # 2. Reshape for Multi-Head / GQA
        # G: (B, T, H_g, D_h) -> (B, H_g, T, D_h)
        g = g.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # V: (B, T, H_v, D_h) -> (B, H_v, T, D_h)
        v = v.view(B, T, self.num_value_heads, self.head_dim).transpose(1, 2)

        logger.info(
            f"MultiHeadGeneticAttention: Reshape took {time.perf_counter() - start_time:.4f}s"
        )
        start_time = time.perf_counter()

        # 3. Compute Genetic Fitness (Attention Scores)
        # Input: (B, H_g, T, D_h)
        # Output: (B, H_g, T, T)
        attn_weights = self.genetic_fitness(g)

        logger.info(
            f"MultiHeadGeneticAttention: Genetic Fitness took {time.perf_counter() - start_time:.4f}s"
        )
        start_time = time.perf_counter()

        # 4. Apply Attention to Values
        # Handle GQA: Repeat V to match G's heads if needed
        if self.num_value_groups > 1:
            # (B, H_v, T, D_h) -> (B, H_v, 1, T, D_h) -> (B, H_v, G, T, D_h) -> (B, H_g, T, D_h)
            v = v.unsqueeze(2).expand(-1, -1, self.num_value_groups, -1, -1)
            v = v.reshape(B, self.num_heads, T, self.head_dim)

        # Weighted sum: (B, H, T, T) @ (B, H, T, D_h) -> (B, H, T, D_h)
        out = torch.matmul(attn_weights, v)

        logger.info(
            f"MultiHeadGeneticAttention: Apply Attention took {time.perf_counter() - start_time:.4f}s"
        )
        start_time = time.perf_counter()

        # 5. Reshape and Project Output
        # (B, H, T, D_h) -> (B, T, H, D_h) -> (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # 6. Gated DeltaNet / Output Projection
        if self.use_gated_deltanet:
            # Apply convolution and gating
            # Conv1d expects (B, D, T)
            gate_input = x.transpose(1, 2)
            gate = self.act(self.conv1d(gate_input)).transpose(1, 2)
            gate = self.gate_proj(gate)
            out = out * torch.sigmoid(gate)

        out = self.out_proj(out)

        logger.info(
            f"MultiHeadGeneticAttention: Output Projection took {time.perf_counter() - start_time:.4f}s"
        )

        return self.dropout(out)
