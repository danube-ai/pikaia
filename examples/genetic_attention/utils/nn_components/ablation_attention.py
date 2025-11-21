"""AblationAttention: Flexible attention mechanism for ablation studies.

This module provides a unified AblationAttention class that can toggle between
different attention mechanisms for systematic ablation studies. It supports:

1. Standard Multi-Head Attention (MHA)
2. Grouped Query Attention (GQA)
3. Multi-Head Latent Attention (MLA)
4. Single-Head Latent Attention (SLA)
5. Genetic sorting on attention weights

The class allows for 8 different ablation combinations by toggling 4 flags:
- use_gqa: Enable Grouped Query Attention
- use_mla: Enable Multi-Head Latent Attention
- use_sla: Enable Single-Head Latent Attention (overrides MLA, uses windowed causal masking)
- use_genetic: Enable genetic sorting on attention weights

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


def compute_genetic_fitness_scores(
    X: torch.Tensor,
    mask: torch.Tensor,
    empty: str = "nan",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute genetic fitness scores using single-GEMM masked weighted output.

    This function calculates fitness scores for genetic attention by performing
    masked weighted aggregation across sequence positions.

    Inputs
    ------
    X : (A, B, C, D)   - input data (batch, heads, seq_len, head_dim)
    mask : (C, C)      - mask/weights (rows are target i, cols are source j)
    empty : 'nan'|'0'|'leave' - behaviour for mask rows with zero sum

    Returns
    -------
    out   : (A, B, C, D)   - final weighted output: out = weight * summed
    weight: (A, B, C, D)   - computed weights per (a,b,i,d)
    phi   : (A, B, C, D)   - per-target means used to compute weights
    """
    # --- shapes ---
    A, B, C, D = X.shape
    assert mask.shape == (C, C), "mask must be shape (C, C)"

    # --- prepare mask on correct device/dtype ---
    m = mask.to(X.dtype).to(X.device)  # m: (C, C)

    # --- counts per-target-channel (row sums of mask) ---
    counts = m.sum(dim=1, keepdim=True)  # counts: (C, 1)

    # --- flatten (A,B,D) into L so we can do one GEMM ---
    # move channel to front and flatten remaining dims:
    # X_perm: (C, A, B, D)
    X_perm = X.permute(2, 0, 1, 3).contiguous()  # X_perm: (C, A, B, D)
    L = A * B * D
    # X_flat: (C, L)
    X_flat = X_perm.reshape(C, L)  # X_flat: (C, L)   <-- big matrix, memory-critical

    # --- single GEMM: mask (C x C) @ X_flat (C x L) -> summed_flat (C x L) ---
    # summed_flat: (C, L)
    summed_flat = m @ X_flat  # summed_flat: (C, L)

    # --- reshape back to (A, B, C, D) ---
    # tmp: (C, A, B, D)
    tmp = summed_flat.reshape(C, A, B, D)  # tmp: (C, A, B, D)
    # summed: (A, B, C, D)
    summed = tmp.permute(1, 2, 0, 3).contiguous()  # summed: (A, B, C, D)

    # --- phi: mean across selected source channels (per-target) ---
    # counts: (C,1) -> counts_safe: (C,1) -> counts_b: (1,1,C,1)
    counts_safe = counts.clamp_min(1.0)  # counts_safe: (C, 1)
    counts_b = counts_safe.view(1, 1, C, 1)  # counts_b: (1, 1, C, 1)
    phi = summed / counts_b  # phi: (A, B, C, D)

    # --- compute weights per your formulas ---
    denom = phi + 0.5  # denom: (A, B, C, D)
    inv_denom = 1.0 / denom  # inv_denom: (A, B, C, D)
    sum_inv = inv_denom.sum(dim=-1, keepdim=True)  # sum_inv: (A, B, C, 1)  (sum over D)
    weight = 1.0 / (denom * sum_inv)  # weight: (A, B, C, D)

    # --- optional handling for mask rows that select nothing ---
    zero_rows = counts.squeeze(1) == 0  # zero_rows: (C,)
    if zero_rows.any():
        idx = zero_rows.nonzero(as_tuple=True)[0]
        if empty == "nan":
            weight[:, :, idx, :] = float("nan")  # weight: (A,B,C,D)
            phi[:, :, idx, :] = float("nan")  # phi: (A,B,C,D)
            summed[:, :, idx, :] = float("nan")  # summed: (A,B,C,D)
        elif empty == "0":
            weight[:, :, idx, :] = 0.0
            phi[:, :, idx, :] = 0.0
            summed[:, :, idx, :] = 0.0
        # 'leave' keeps whatever the current numerics are (counts were clamped)

    # --- final output: multiply weight by the mask-weighted sum (original-input-based sum) ---
    out = weight * summed  # out: (A, B, C, D)

    # --- clean up temporary view references to help memory reclaiming in long-running contexts ---
    # (no explicit del needed, but avoids holding extra references)
    # return results
    return out, weight, phi


class AblationAttention(nn.Module):
    """
    Flexible attention mechanism for ablation studies.

    Supports toggling between different attention variants:
    - Standard MHA (default)
    - GQA (Grouped Query Attention)
    - MLA (Multi-Head Latent Attention)
    - SLA (Single-Head Latent Attention)
    - Genetic sorting on attention weights

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
        use_genetic: Enable genetic sorting on attention weights
        qk_norm: Whether to apply RMS normalization to Q and K
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
            self.mla_up_v = nn.Linear(
                self.latent_dim, self.num_kv_heads * self.head_dim
            )

            # Layer norms for MLA
            if not use_genetic:
                self.ln_q = nn.LayerNorm(embed_dim)
                self.ln_k = nn.LayerNorm(self.num_kv_heads * self.head_dim)
            self.ln_v = nn.LayerNorm(self.num_kv_heads * self.head_dim)
        else:
            # Standard projections
            if not use_genetic:
                self.w_q = nn.Linear(embed_dim, embed_dim, bias=bias)
                self.w_k = nn.Linear(
                    embed_dim, self.num_kv_heads * self.head_dim, bias=bias
                )
            self.w_v = nn.Linear(
                embed_dim, self.num_kv_heads * self.head_dim, bias=bias
            )

        # QK RMS normalization (only for non-genetic)
        if qk_norm and not use_genetic:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout_layer = nn.Dropout(dropout)

    def _compute_standard_attention(
        self,
        x: torch.Tensor,
        batch_size: int,
        seq_len: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute standard attention mechanisms (MHA, GQA, MLA, SLA)."""
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
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )

        # 3. Apply QK normalization if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 4. Handle GQA: expand K and V to match number of query heads
        if self.use_gqa:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        # 5. Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 6. Apply masking
        if self.use_sla:
            # SLA: Use windowed causal mask
            sliding_mask = self._create_sliding_window_mask(seq_len, device=x.device)
            scores = scores.masked_fill(sliding_mask, float("-inf"))
        else:
            # Non-SLA: Use full causal mask
            if self.is_causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
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
            # New genetic attention formulation: only use V
            # 1. Project inputs to V only
            if self.use_mla:
                # MLA: Compress inputs then project to V
                compressed = self.mla_down(x)
                v = self.ln_v(self.mla_up_v(compressed))
            else:
                # Standard V projection
                v = self.w_v(x)

            # 2. Reshape V for multi-head
            v = (
                v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )  # (B, num_kv_heads, T, D_h)

            # 3. Handle GQA: expand V to match number of heads
            if self.use_gqa:
                v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
                v = v.reshape(
                    batch_size, self.num_heads, seq_len, self.head_dim
                ).contiguous()

            # 4. Create mask based on SLA
            if self.use_sla:
                # SLA: Use windowed causal mask
                mask = (
                    self._create_sliding_window_mask(seq_len, device=x.device)
                    .squeeze(0)
                    .squeeze(0)
                )
            else:
                # Non-SLA: Use full causal mask
                mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1,
                )

            # Scale v to [0,1] globally for genetic algorithm
            v_min = v.amin(dim=[1, 2, 3], keepdim=True)  # (B, 1, 1, 1)
            v_max = v.amax(dim=[1, 2, 3], keepdim=True)  # (B, 1, 1, 1)
            v_scaled = (v - v_min) / (v_max - v_min + 1e-8)  # (B, H, T, D_h)

            # Use single GEMM to compute local gene_fitness
            m = (~mask).float()  # mask with 1 for valid positions
            out, gene_fitness, phi = compute_genetic_fitness_scores(v_scaled, m)

            # gene_fitness: (B, H, T, D_h)

            # Compute org_fitness: org_fitness[b,h,i,j] = v_scaled[b,h,j] @ gene_fitness[b,h,i]
            org_fitness = torch.einsum(
                "b h j d, b h i d -> b h i j", v_scaled, gene_fitness
            )

            # Mask invalid positions to -inf
            org_fitness = org_fitness.masked_fill(
                mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

            # Apply softmax to org_fitness
            attn_weights = F.softmax(org_fitness, dim=-1)  # (B, H, T, T)

            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)  # (B, H, T, D_h)

        else:
            attn_output = self._compute_standard_attention(
                x, batch_size, seq_len, attention_mask
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
