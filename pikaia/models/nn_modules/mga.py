"""
Multihead Genetic Attention Module
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadGeneticAttention(nn.Module):
    """
    Multihead Genetic Attention module implementing multihead attention with
    extensions for Grouped Query Attention (GQA) and Multi-Head Latent Attention (MLA).

    This module provides efficient attention mechanisms by supporting fewer key-value
    heads (GQA) and optional input projection compression (MLA).

    Parameters:
        embed_dim (int):
            Total dimension of the model.
        n_heads (int):
            Number of attention heads for queries.
        n_kv_heads (int, optional):
            Number of attention heads for keys and values (GQA). Defaults to n_heads.
        in_proj_dim (int, optional):
            Dimension to project input to before attention (MLA). If None, no projection.
        dropout (float):
            Dropout probability. Defaults to 0.0.
        bias (bool):
            Whether to use bias in linear projections. Defaults to True.

    Input:
        x (torch.Tensor):
            Input tensor of shape (batch_size, seq_len, embed_dim).

    Output:
        torch.Tensor:
            Output tensor of shape (batch_size, seq_len, embed_dim).
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        in_proj_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.in_proj_dim = in_proj_dim
        self.dropout = dropout
        self.bias = bias

        # Effective dimension after optional input projection
        effective_dim = in_proj_dim if in_proj_dim is not None else embed_dim

        # Head dimension
        self.head_dim = effective_dim // n_heads
        if effective_dim % n_heads != 0:
            raise ValueError(
                f"effective_dim ({effective_dim}) must be divisible "
                f"by n_heads ({n_heads})"
            )

        # Input projection for MLA
        if in_proj_dim is not None:
            self.input_proj = nn.Linear(embed_dim, in_proj_dim, bias=bias)
        else:
            self.input_proj = None

        # Query projection
        self.q_proj = nn.Linear(effective_dim, n_heads * self.head_dim, bias=bias)
        # Key and Value projections (shared for GQA)
        self.k_proj = nn.Linear(
            effective_dim, self.n_kv_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            effective_dim, self.n_kv_heads * self.head_dim, bias=bias
        )

        # Output projection
        self.out_proj = nn.Linear(effective_dim, embed_dim, bias=bias)

        # RMS normalization for Q and K
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        disable_genetic: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of MultiheadGeneticAttention.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, seq_len, embed_dim).
            attn_mask (torch.Tensor, optional):
                Attention mask of shape (batch_size, seq_len) where True indicates
                valid tokens and False indicates padding tokens. Defaults to None.
            disable_genetic (bool):
                If True, disables the genetic fitness computation and uses standard
                attention. Defaults to False.

        Returns:
            torch.Tensor:
                Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.shape

        # 1. Optional input projection (MLA)
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, in_proj_dim)
        if self.input_proj is not None:
            x = self.input_proj(x)

        # 2. Project to queries, keys, values
        # q,k,v -> (batch_size, n_heads/n_kv_heads, seq_len, head_dim)
        q: torch.Tensor = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k: torch.Tensor = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v: torch.Tensor = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # 3. Handle GQA: repeat K and V for grouped heads
        # k,v -> (batch_size, n_heads, seq_len, head_dim)
        if self.n_kv_heads != self.n_heads:
            num_groups = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_groups, dim=1)
            v = v.repeat_interleave(num_groups, dim=1)

        # 4. Apply QK normalization
        # q,k -> (batch_size, n_heads, seq_len, head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 5. Get genetic gene fitness values or use standard values
        if disable_genetic:
            v_genetic = v
        else:
            # gene_fitness -> (batch_size, n_heads, head_dim)
            gene_fitness = self._compute_gene_fitness(v, attn_mask)
            # v_genetic -> (batch_size, n_heads, seq_len, head_dim)
            v_genetic = v * gene_fitness.unsqueeze(-2)

        # 7. Attention computation
        # attn_weights -> (batch_size, n_heads, seq_len, seq_len)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            # attn_mask: (batch_size, seq_len) where True = valid token, False = padding
            # Expand to (batch_size, 1, 1, seq_len) for broadcasting
            expanded_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~expanded_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        if self.dropout > 0.0:
            attn_weights = F.dropout(
                attn_weights, p=self.dropout, training=self.training
            )

        # 8. Apply attention to values
        # attn_output -> (batch_size, n_heads, seq_len, head_dim)
        attn_output = attn_weights @ v_genetic

        # 9. Reshape and project output
        # attn_output -> (batch_size, seq_len, embed_dim)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.n_heads * self.head_dim)
        )
        output = self.out_proj(attn_output)

        return output

    def _compute_gene_fitness(
        self, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gene fitness scores using genetic algorithm formulation.

        Args:
            v (torch.Tensor):
                Value matrix of shape (batch_size, n_heads, seq_len, head_dim)
            attn_mask (torch.Tensor, optional):
                Attention mask of shape (batch_size, seq_len)

        Returns:
            torch.Tensor:
                Gene fitness scores of shape (batch_size, n_heads, head_dim)
        """
        batch_size, n_heads, seq_len, head_dim = v.shape

        # Normalize values from 0 to 1 using sigmoid
        # v_normalized: (batch_size, n_heads, seq_len, head_dim)
        v_normalized = torch.sigmoid(v)

        # Reshape for genetic computation: (batch_size * n_heads, seq_len, head_dim)
        v_flat = v_normalized.reshape(batch_size * n_heads, seq_len, head_dim)

        if attn_mask is not None:
            # attn_mask: (batch_size, seq_len) -> (batch_size, 1, seq_len)
            attn_mask_expanded = attn_mask.unsqueeze(1)
            # Repeat for n_heads: (batch_size, n_heads, seq_len)
            attn_mask_heads = attn_mask_expanded.repeat(1, n_heads, 1)
            # Flatten: (batch_size * n_heads, seq_len)
            attn_mask_flat = attn_mask_heads.view(batch_size * n_heads, seq_len)
        else:
            attn_mask_flat = None

        # For each batch*head, compute gene means over valid organisms (seq positions)
        # gene_means: (batch_size * n_heads, head_dim)
        if attn_mask_flat is not None:
            # Mask out invalid positions by setting them to 0, then divide by sum of mask
            masked_v = v_flat * attn_mask_flat.unsqueeze(
                -1
            )  # (batch_size*n_heads, seq_len, head_dim)
            gene_means = masked_v.sum(dim=1) / attn_mask_flat.sum(
                dim=1, keepdim=True
            ).clamp(min=1)  # (batch_size*n_heads, head_dim)
        else:
            gene_means = v_flat.mean(dim=1)  # (batch_size*n_heads, head_dim)

        # Compute gene fitness following the genetic formulation
        # γ_j* = (∑_{s=1}^m (Φ̃_j + 1/2)/(Φ̃_s + 1/2) )^{-1}
        denom = gene_means + 0.5  # (batch_size*n_heads, head_dim)
        sum_ratios = (denom.unsqueeze(-1) / denom.unsqueeze(-2)).sum(
            dim=-1
        )  # (batch_size*n_heads, head_dim)
        gene_fitness = 1.0 / sum_ratios  # (batch_size*n_heads, head_dim)

        # Reshape back: (batch_size, n_heads, head_dim)
        gene_fitness = gene_fitness.view(batch_size, n_heads, head_dim)

        return gene_fitness
