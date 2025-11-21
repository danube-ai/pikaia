"""AblationAttention: Flexible attention mechanism for ablation studies.

This module provides a unified AblationAttention class that can toggle between
different attention mechanisms for systematic ablation studies. It supports:

1. Standard MHA (Multi-Head Attention)
2. Grouped Query Attention (GQA)
3. Multi-Head Latent Attention (MLA)
4. Single-Head Latent Attention (SLA)
5. Genetic Attention (simplified genetic fitness-based attention)

The class allows for 8 different ablation combinations by toggling 4 flags:
- use_gqa: Enable Grouped Query Attention
- use_mla: Enable Multi-Head Latent Attention
- use_sla: Enable Single-Head Latent Attention (overrides MLA, uses windowed causal masking)
- use_genetic: Enable simplified genetic attention
- qk_norm: Whether to apply RMS normalization to Q and K (only for non-genetic)

Masking logic:
- SLA variants: Windowed causal mask (attention limited to window_size past positions)
- Non-SLA variants: Full causal mask (attention to all past positions)

Default configuration provides standard multi-head causal autoregressive attention with QK RMS normalization.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AblationAttention(nn.Module):
    """
    Flexible attention mechanism for ablation studies.

    Supports toggling between different attention variants:
    - Standard MHA (default)
    - GQA (Grouped Query Attention)
    - MLA (Multi-Head Latent Attention)
    - SLA (Single-Head Latent Attention)
    - Genetic Attention (simplified genetic fitness-based attention)

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of attention heads
        window_size: Size of the sliding window for attention
        dropout: Dropout probability
        bias: Whether to use bias in projections
        use_gqa: Enable Grouped Query Attention (fewer KV heads)
        num_kv_heads: Number of key-value heads for GQA (default: num_heads // 2 when use_gqa=True, ignored when use_gqa=False)
        use_mla: Enable Multi-Head Latent Attention (input compression)
        latent_dim: Dimension of latent space for MLA (ignored if use_mla=False)
        use_sla: Enable Single-Head Latent Attention (overrides MLA, sets num_heads=1)
        use_genetic: Enable simplified genetic attention (uses O', G', V projections with fitness modulation)
        qk_norm: Whether to apply RMS normalization to Q and K (only for non-genetic)
        is_causal: Whether to use causal masking
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        dropout: float = 0.0,
        bias: bool = True,
        use_gqa: bool = False,
        num_kv_heads: Optional[int] = None,
        use_mla: bool = False,
        latent_dim: Optional[int] = None,
        use_sla: bool = False,
        use_genetic: bool = False,
        qk_norm: bool = True,
        is_causal: bool = True,
    ):
        super().__init__()

        # Validate inputs
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # SLA overrides MLA and sets num_heads to 1
        if use_sla:
            use_mla = True
            num_heads = 1

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.dropout = dropout
        self.bias = bias
        self.use_gqa = use_gqa
        self.use_mla = use_mla
        self.use_sla = use_sla
        self.use_genetic = use_genetic
        self.qk_norm = qk_norm
        self.is_causal = is_causal

        # GQA configuration
        if use_gqa:
            self.num_kv_heads = (
                num_kv_heads if num_kv_heads is not None else num_heads // 2
            )
            assert num_heads % self.num_kv_heads == 0, (
                "num_heads must be divisible by num_kv_heads"
            )
            self.num_kv_groups = num_heads // self.num_kv_heads
        else:
            self.num_kv_heads = num_heads
            self.num_kv_groups = 1

        # MLA/SLA configuration
        if use_mla:
            self.latent_dim = (
                latent_dim if latent_dim is not None else int(embed_dim * 0.5)
            )

        # Projections
        if use_mla:
            # MLA: Low-rank compression for inputs
            self.mla_down = nn.Linear(embed_dim, self.latent_dim)
            if not use_genetic:
                self.mla_up_q = nn.Linear(self.latent_dim, embed_dim)
                self.mla_up_k = nn.Linear(
                    self.latent_dim, self.num_kv_heads * self.head_dim
                )
            else:
                # Genetic: need both Q and K projections
                self.mla_up_q = nn.Linear(self.latent_dim, embed_dim)
                self.mla_up_k = nn.Linear(self.latent_dim, embed_dim)
            self.mla_up_v = nn.Linear(
                self.latent_dim, self.num_kv_heads * self.head_dim
            )

            # Layer norms for MLA
            if not use_genetic:
                self.ln_q = nn.LayerNorm(embed_dim)
                self.ln_k = nn.LayerNorm(self.num_kv_heads * self.head_dim)
            else:
                # Genetic: reuse Q and K norms
                self.ln_q = nn.LayerNorm(embed_dim)
                self.ln_k = nn.LayerNorm(embed_dim)
            self.ln_v = nn.LayerNorm(self.num_kv_heads * self.head_dim)
        else:
            # Standard projections
            if not use_genetic:
                self.w_q = nn.Linear(embed_dim, embed_dim, bias=bias)
                self.w_k = nn.Linear(
                    embed_dim, self.num_kv_heads * self.head_dim, bias=bias
                )
            else:
                # Genetic: need both Q and K projections at full dimension
                self.w_q = nn.Linear(embed_dim, embed_dim, bias=bias)
                self.w_k = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.w_v = nn.Linear(
                embed_dim, self.num_kv_heads * self.head_dim, bias=bias
            )

        # QK RMS normalization (only for non-genetic)
        if qk_norm and not use_genetic:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

        # For genetic: Q and K norms (for both MLA and non-MLA)
        if use_genetic and qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout_layer = nn.Dropout(dropout)

        # For genetic attention: store fitness values
        self._gene_fitness = None

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        seq_len: int,
        attention_mask: Optional[torch.Tensor] = None,
        use_genetic: bool = False,
    ) -> torch.Tensor:
        """Compute attention with optional genetic fitness modulation."""
        # For genetic attention, q and k have full embed_dim (num_heads projections)
        # For standard attention, k and v may have fewer heads (GQA)
        num_k_heads = self.num_heads if use_genetic else self.num_kv_heads
        num_v_heads = self.num_kv_heads  # V always follows KV head pattern

        # 1. Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_k_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_v_heads, self.head_dim).transpose(1, 2)

        # 2. Apply QK normalization if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 3. Handle GQA: expand K and V to match number of query heads
        if self.use_gqa and not use_genetic:
            # Only expand for standard attention with GQA
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        elif use_genetic and self.use_gqa:
            # For genetic GQA, we need to expand V but K is already full-size
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        # 4. Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 5. Apply masking
        if self.use_sla:
            # SLA: Use windowed causal mask
            sliding_mask = self._create_sliding_window_mask(seq_len, device=q.device)
            scores = scores.masked_fill(sliding_mask, float("-inf"))
        else:
            # Non-SLA: Use full causal mask
            if self.is_causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                    diagonal=1,
                )
                scores = scores.masked_fill(
                    causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(attention_mask, float("-inf"))

        # 6. Genetic fitness modulation (if enabled)
        if use_genetic:
            # Compute gene fitness scores using PikaiaModel approach
            # Apply sigmoid to get values in [0, 1] range - sigmoid(-inf) = 0 naturally
            sigmoid_scores = torch.sigmoid(scores)  # (B, H, T, T)

            # Compute gene means: average expression of each gene across all positions
            gene_means = sigmoid_scores.mean(dim=-2)  # (B, H, T)

            # Compute fitness
            denom = gene_means + 0.5  # (B, H, T)
            sum_inv_denom = torch.sum(1.0 / denom, dim=-1, keepdim=True)  # (B, H, 1)
            gene_fitness = 1.0 / (denom * sum_inv_denom)  # (B, H, T)

            # Clamp fitness values to prevent extreme modulation that causes NaN
            gene_fitness = torch.clamp(gene_fitness, min=0.1, max=10.0)

            # Store fitness for modulation before softmax
            self._gene_fitness = gene_fitness

            # Modulate attention scores by broadcasting fitness before softmax
            # This biases attention towards positions with higher genetic fitness
            scores = scores * self._gene_fitness.unsqueeze(-2)

        # 7. Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # 8. Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for ablation attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # All modalities now use consistent masking: full causal unless SLA, then windowed causal
        batch_size, seq_len, embed_dim = x.shape

        if self.use_genetic:
            # Simplified genetic attention: uses O' (Q) and G' (K) projections with fitness modulation
            # 1. Project inputs to O', G', V (O' uses Q, G' uses K)
            if self.use_mla:
                # MLA: Compress inputs then project (O' uses Q, G' uses K)
                compressed = self.mla_down(x)
                o_prime = self.ln_q(self.mla_up_q(compressed))  # O' = Q
                g_prime = self.ln_k(self.mla_up_k(compressed))  # G' = K
                v = self.ln_v(self.mla_up_v(compressed))
            else:
                # Standard projections (O' uses Q, G' uses K)
                o_prime = self.w_q(x)  # O' = Q
                g_prime = self.w_k(x)  # G' = K
                v = self.w_v(x)

            # 2. Apply RMS normalization to O' and G' if enabled
            if self.qk_norm:
                # Reshape for per-head normalization
                o_prime_reshaped = o_prime.view(
                    batch_size, seq_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                g_prime_reshaped = g_prime.view(
                    batch_size, seq_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                o_prime_reshaped = self.q_norm(o_prime_reshaped)
                g_prime_reshaped = self.k_norm(g_prime_reshaped)
                # Reshape back
                o_prime = o_prime_reshaped.transpose(1, 2).reshape(
                    batch_size, seq_len, self.embed_dim
                )
                g_prime = g_prime_reshaped.transpose(1, 2).reshape(
                    batch_size, seq_len, self.embed_dim
                )

            attn_output = self._compute_attention(
                o_prime,
                g_prime,
                v,
                batch_size,
                seq_len,
                attention_mask,
                use_genetic=True,
            )

        else:
            # Standard attention: compute Q, K, V projections
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

            attn_output = self._compute_attention(
                q, k, v, batch_size, seq_len, attention_mask, use_genetic=False
            )

        # 9. Reshape and project output
        # (B, H, T, D_h) -> (B, T, H, D_h) -> (B, T, D)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, embed_dim
        )

        # Final projection
        output = self.out_proj(attn_output)
        output = self.dropout_layer(output)

        return output

    def _create_sliding_window_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create a sliding window attention mask.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Boolean mask of shape (1, 1, seq_len, seq_len)
            where True indicates positions to mask out.
        """
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)

        # Allow attention within window_size to the left and current position
        mask = (col_idx > row_idx) | (row_idx - col_idx >= self.window_size)

        return mask.unsqueeze(0).unsqueeze(0)


def create_ablation_configs(
    embed_dim: int, num_heads: int, window_size: int
) -> list[dict]:
    """
    Create all 8 ablation study configurations.

    Returns a list of configuration dictionaries for the 8 ablation combinations:
    1. Standard MHA
    2. MHA + Genetic
    3. GQA
    4. GQA + Genetic
    5. MLA
    6. MLA + Genetic
    7. SLA
    8. SLA + Genetic

    Args:
        embed_dim: Model embedding dimension
        num_heads: Number of attention heads
        window_size: Attention window size

    Returns:
        List of configuration dictionaries
    """
    base_config = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "window_size": window_size,
        "dropout": 0.0,
        "bias": True,
        "qk_norm": True,
        "is_causal": True,
    }

    configs = [
        # 1. Standard MHA
        {
            **base_config,
            "use_gqa": False,
            "use_mla": False,
            "use_sla": False,
            "use_genetic": False,
        },
        # 2. MHA + Genetic
        {
            **base_config,
            "use_gqa": False,
            "use_mla": False,
            "use_sla": False,
            "use_genetic": True,
        },
        # 3. GQA
        {
            **base_config,
            "use_gqa": True,
            "num_kv_heads": 3,
            "use_mla": False,
            "use_sla": False,
            "use_genetic": False,
        },
        # 4. GQA + Genetic
        {
            **base_config,
            "use_gqa": True,
            "num_kv_heads": 3,
            "use_mla": False,
            "use_sla": False,
            "use_genetic": True,
        },
        # 5. MLA
        {
            **base_config,
            "use_gqa": False,
            "use_mla": True,
            "latent_dim": embed_dim // 2,
            "use_sla": False,
            "use_genetic": False,
        },
        # 6. MLA + Genetic
        {
            **base_config,
            "use_gqa": False,
            "use_mla": True,
            "latent_dim": embed_dim // 2,
            "use_sla": False,
            "use_genetic": True,
        },
        # 7. SLA
        {
            **base_config,
            "use_gqa": False,
            "use_mla": False,
            "use_sla": True,
            "latent_dim": embed_dim // 2,
            "use_genetic": False,
        },
        # 8. SLA + Genetic
        {
            **base_config,
            "use_gqa": False,
            "use_mla": False,
            "use_sla": True,
            "latent_dim": embed_dim // 2,
            "use_genetic": True,
        },
    ]

    return configs


def get_config_name(config: dict) -> str:
    """
    Get a human-readable name for an ablation configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration name string
    """
    name_parts = []

    if config.get("use_sla", False):
        name_parts.append("SLA")
    elif config.get("use_mla", False):
        name_parts.append("MLA")
    elif config.get("use_gqa", False):
        name_parts.append("GQA")
    else:
        name_parts.append("MHA")

    if config.get("use_genetic", False):
        name_parts.append("Genetic")

    return " + ".join(name_parts) if name_parts else "Standard"
