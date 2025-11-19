"""Multi-Head Latent Attention (MLA) implementation with optional Sliding Window Attention."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention mechanism with optional Sliding Window Attention.

    MLA compresses the key-value representations into a lower-dimensional latent space
    before expanding them back for attention computation. This reduces memory usage
    while maintaining expressiveness through the latent bottleneck.

    The architecture follows:
    1. Project input to latent space (compression)
    2. Expand latent to keys and values
    3. Compute queries directly from input
    4. Perform standard multi-head attention

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of attention heads.
        latent_dim (int): Dimension of the latent compression space.
        dropout (float): Dropout probability. Default: 0.0
        bias (bool): Whether to use bias in projections. Default: True
        use_sliding_window (bool): Whether to use sliding window attention. Default: False
        window_size (Optional[int]): Size of the sliding window. Required if use_sliding_window=True.
        qk_norm (bool): Whether to apply layer normalization to queries and keys. Default: False
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        latent_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
        use_sliding_window: bool = False,
        window_size: Optional[int] = None,
        qk_norm: bool = False,
        is_causal: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.is_causal = is_causal

        if use_sliding_window:
            assert window_size is not None, (
                "window_size must be specified when use_sliding_window=True"
            )

        # Latent compression projection
        self.latent_proj = nn.Linear(embed_dim, latent_dim, bias=bias)
        self.latent_norm = nn.LayerNorm(latent_dim)

        # QK normalization (optional)
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

        # Expand latent to keys and values
        self.kv_expansion = nn.Linear(latent_dim, 2 * embed_dim, bias=bias)

        # Query projection (standard)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for multi-head latent attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask (Optional[torch.Tensor]): Mask of shape (batch_size, seq_len, seq_len)
                or (batch_size, num_heads, seq_len, seq_len). True values are masked out.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Query projection: (batch_size, seq_len, embed_dim)
        q = self.q_proj(x)
        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)

        # Compress to latent space: (batch_size, seq_len, latent_dim)
        latent = self.latent_proj(x)
        latent = self.latent_norm(latent)

        # Expand latent to KV: (batch_size, seq_len, 2 * embed_dim)
        kv = self.kv_expansion(latent)

        # Split into K and V: each (batch_size, seq_len, embed_dim)
        k, v = kv.chunk(2, dim=-1)

        # Reshape K to (batch_size, seq_len, num_heads, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)

        # Reshape V to (batch_size, seq_len, num_heads, head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        v = v.transpose(1, 2)

        # Apply QK normalization if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Compute attention scores: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask if enabled
        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        # Apply sliding window mask if enabled
        if self.use_sliding_window:
            sliding_mask = self._create_sliding_window_mask(seq_len, device=x.device)
            scores = scores.masked_fill(sliding_mask, float("-inf"))

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                # Expand to (batch_size, num_heads, seq_len, seq_len)
                attention_mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(attention_mask, float("-inf"))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values: (batch_size, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Concatenate heads: (batch_size, seq_len, embed_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)

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
            seq_len (int): Sequence length
            device (torch.device): Device to create mask on

        Returns:
            torch.Tensor: Boolean mask of shape (1, 1, seq_len, seq_len)
                where True indicates positions to mask out.
        """
        # Create position indices
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)

        # Create mask: True where distance > window_size or future positions
        # Allow attention within window_size to the left and current position
        window_size_val = self.window_size if self.window_size is not None else seq_len
        mask = (col_idx > row_idx) | (row_idx - col_idx >= window_size_val)

        # Expand to (1, 1, seq_len, seq_len) for broadcasting
        return mask.unsqueeze(0).unsqueeze(0)
