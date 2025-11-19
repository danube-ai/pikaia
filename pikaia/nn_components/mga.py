from typing import Optional

import torch
import torch.nn as nn


class GeneticFitness(nn.Module):
    """
    Computes genetic fitness scores (attention weights) from a population matrix.

    This module implements the "fixed organism, balanced gene, dominant strategy"
    adapted for attention mechanisms. It ensures that masking is applied *before*
    genetic sorting to prevent information leakage in autoregressive settings.
    """

    def __init__(self, window_size: Optional[int] = None):
        super().__init__()
        self.window_size = window_size

    def forward(self, population_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            population_matrix: Shape (batch_size, num_heads, seq_length, head_dim)
                             or (batch_size, seq_length, head_dim)

        Returns:
            normalized_fitness: Shape (batch_size, num_heads, seq_length, seq_length)
                              or (batch_size, seq_length, seq_length)
        """
        # Handle input shapes
        if population_matrix.dim() == 3:
            # (B, T, D) -> (B, 1, T, D)
            population_matrix = population_matrix.unsqueeze(1)

        B, H, T, D = population_matrix.shape
        device = population_matrix.device

        # 1. Create Mask
        # Start with full causal mask
        mask = torch.tril(torch.ones(T, T, device=device))

        # Apply Sliding Window Attention (SWA) if configured
        if self.window_size is not None:
            # Keep only the band within window_size
            # tril(ones) - tril(ones, diagonal=-window_size)
            window_mask = torch.tril(
                torch.ones(T, T, device=device), diagonal=0
            ) - torch.tril(torch.ones(T, T, device=device), diagonal=-self.window_size)
            mask = mask * window_mask

        # Reshape mask for broadcasting: (1, 1, T, T, 1)
        # We need to apply it to the expanded population
        mask_expanded = mask.view(1, 1, T, T, 1)

        # 2. Expand Population and Apply Mask BEFORE computation
        # We broadcast the population to create a "view" for each token.
        # population_matrix: (B, H, T, D) -> unsqueeze(2) -> (B, H, 1, T, D)
        # mask_expanded: (1, 1, T, T, 1)
        # Result: (B, H, T, T, D) where dim 2 is "viewing token" and dim 3 is "organism"

        masked_pop = population_matrix.unsqueeze(2) * mask_expanded

        # 3. Compute means over the valid organisms for each token
        # Sum over dim 3 (organisms) -> (B, H, T, D)
        active_counts = mask.sum(dim=1).view(1, 1, T, 1)
        active_counts = active_counts.clamp(min=1)  # Avoid div by zero

        gene_means = masked_pop.sum(dim=3) / active_counts

        # 4. Standard genetic fitness computation
        # Shape: (B, H, T, D)
        denom = gene_means + 0.5
        sum_inv_denom = torch.sum(1 / denom, dim=-1, keepdim=True)
        gene_fitness = 1 / (denom * sum_inv_denom)

        # 5. Compute raw fitness scores using the MASKED population
        # masked_pop: (B, H, T, T, D)
        # gene_fitness: (B, H, T, D) -> unsqueeze(3) -> (B, H, T, 1, D)
        # Product sums over D -> (B, H, T, T)

        org_fitness = (masked_pop * gene_fitness.unsqueeze(3)).sum(dim=-1)

        # 6. Normalize each row to sum to 1
        row_sums = org_fitness.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        normalized_fitness = org_fitness / row_sums

        return normalized_fitness


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
        num_value_heads: Optional[int] = None,
        window_size: Optional[int] = None,
        use_mla: bool = False,
        mla_compression_dim: Optional[int] = None,
        use_gated_deltanet: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
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

        # 2. Reshape for Multi-Head / GQA
        # G: (B, T, H_g, D_h) -> (B, H_g, T, D_h)
        g = g.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # V: (B, T, H_v, D_h) -> (B, H_v, T, D_h)
        v = v.view(B, T, self.num_value_heads, self.head_dim).transpose(1, 2)

        # 3. Compute Genetic Fitness (Attention Scores)
        # Input: (B, H_g, T, D_h)
        # Output: (B, H_g, T, T)
        attn_weights = self.genetic_fitness(g)

        # 4. Apply Attention to Values
        # Handle GQA: Repeat V to match G's heads if needed
        if self.num_value_groups > 1:
            # (B, H_v, T, D_h) -> (B, H_v, 1, T, D_h) -> (B, H_v, G, T, D_h) -> (B, H_g, T, D_h)
            v = v.unsqueeze(2).expand(-1, -1, self.num_value_groups, -1, -1)
            v = v.reshape(B, self.num_heads, T, self.head_dim)

        # Weighted sum: (B, H, T, T) @ (B, H, T, D_h) -> (B, H, T, D_h)
        out = torch.matmul(attn_weights, v)

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
        return self.dropout(out)
