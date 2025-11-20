"""SmolLM 2 architecture (~135M parameters) with modular attention mechanisms."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU activation function used in modern transformers.

    SwiGLU(x) = Swish(xW) ⊗ (xV)
    where ⊗ is element-wise multiplication.

    Args:
        dim (int): Input dimension
        hidden_dim (int): Hidden dimension (typically 4 * dim)
        bias (bool): Whether to use bias. Default: False
    """

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w = nn.Linear(dim, hidden_dim, bias=bias)
        self.v = nn.Linear(dim, hidden_dim, bias=bias)
        self.proj = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        swish_output = F.silu(self.w(x))
        gated_output = swish_output * self.v(x)
        return self.proj(gated_output)


class TransformerBlock(nn.Module):
    """
    Single transformer block with configurable normalization strategy.

    Supports three normalization strategies:
    - 'pre': Pre-normalization (Llama 3 8B style) - norm before attention/FFN
    - 'post': Post-normalization (OLMo 2 7B style) - norm after attention/FFN but before residual
    - 'sandwich': Sandwich normalization - both pre and post norm

    Args:
        embed_dim (int): Embedding dimension
        attention_module (nn.Module): Attention module to use (MHA, GQA, MLA, etc.)
        ffn_hidden_dim (int): Hidden dimension for feedforward network
        dropout (float): Dropout probability. Default: 0.0
        use_bias (bool): Whether to use bias in linear layers. Default: False
        norm_strategy (str): Normalization strategy ('pre', 'post', 'sandwich'). Default: 'sandwich'
    """

    def __init__(
        self,
        embed_dim: int,
        attention_module: nn.Module,
        ffn_hidden_dim: int,
        dropout: float = 0.0,
        use_bias: bool = False,
        norm_strategy: str = "sandwich",
    ):
        super().__init__()

        assert norm_strategy in ["pre", "post", "sandwich"], (
            f"norm_strategy must be 'pre', 'post', or 'sandwich', got '{norm_strategy}'"
        )

        self.norm_strategy = norm_strategy

        # Normalization layers (create based on strategy)
        if norm_strategy in ["pre", "sandwich"]:
            self.attn_norm_pre = nn.RMSNorm(embed_dim)
            self.ffn_norm_pre = nn.RMSNorm(embed_dim)

        if norm_strategy in ["post", "sandwich"]:
            self.attn_norm_post = nn.RMSNorm(embed_dim)
            self.ffn_norm_post = nn.RMSNorm(embed_dim)

        # Attention module (injected)
        self.attention = attention_module

        # Feedforward network with SwiGLU
        self.ffn = SwiGLU(embed_dim, ffn_hidden_dim, bias=use_bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask (Optional[torch.Tensor]): Attention mask

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Self-attention with residual connection
        if self.norm_strategy == "pre":
            # Pre-norm: norm -> attention -> residual
            attn_input = self.attn_norm_pre(x)
            attn_output = self.attention(attn_input, attention_mask=attention_mask)
            x = x + self.dropout(attn_output)
        elif self.norm_strategy == "post":
            # Post-norm: attention -> norm -> residual
            attn_output = self.attention(x, attention_mask=attention_mask)
            attn_output = self.attn_norm_post(attn_output)
            x = x + self.dropout(attn_output)
        else:  # sandwich
            # Sandwich: norm -> attention -> norm -> residual
            attn_input = self.attn_norm_pre(x)
            attn_output = self.attention(attn_input, attention_mask=attention_mask)
            attn_output = self.attn_norm_post(attn_output)
            x = x + self.dropout(attn_output)

        # Feedforward with residual connection
        if self.norm_strategy == "pre":
            # Pre-norm: norm -> ffn -> residual
            ffn_input = self.ffn_norm_pre(x)
            ffn_output = self.ffn(ffn_input)
            x = x + self.dropout(ffn_output)
        elif self.norm_strategy == "post":
            # Post-norm: ffn -> norm -> residual
            ffn_output = self.ffn(x)
            ffn_output = self.ffn_norm_post(ffn_output)
            x = x + self.dropout(ffn_output)
        else:  # sandwich
            # Sandwich: norm -> ffn -> norm -> residual
            ffn_input = self.ffn_norm_pre(x)
            ffn_output = self.ffn(ffn_input)
            ffn_output = self.ffn_norm_post(ffn_output)
            x = x + self.dropout(ffn_output)

        return x


class SmolLM(nn.Module):
    """
    SmolLM 2 architecture (~135M parameters) with modular attention mechanisms.

    This architecture follows modern LLM design principles:
    - RMSNorm for layer normalization
    - SwiGLU for feedforward activation
    - Pre-normalization
    - Modular attention (swap between MHA, GQA, MLA, etc.)
    - Rotary positional embeddings (RoPE) support via attention modules

    Default configuration targets ~135M parameters:
    - vocab_size: 49152
    - embed_dim: 576
    - num_layers: 30
    - num_heads: 9
    - ffn_hidden_dim: 1536 (2.67x embed_dim)

    Args:
        vocab_size (int): Vocabulary size. Default: 49152
        embed_dim (int): Embedding dimension. Default: 576
        num_layers (int): Number of transformer layers. Default: 30
        attention_module (nn.Module): Attention module to use in each block
        ffn_hidden_dim (int): Hidden dimension for FFN. Default: 1536
        max_seq_len (int): Maximum sequence length. Default: 2048
        dropout (float): Dropout probability. Default: 0.0
        use_bias (bool): Whether to use bias in linear layers. Default: False
        tie_embeddings (bool): Whether to tie input/output embeddings. Default: True
        norm_strategy (str): Normalization strategy ('pre', 'post', 'sandwich'). Default: 'pre'
    """

    def __init__(
        self,
        vocab_size: int = 49152,
        embed_dim: int = 576,
        num_layers: int = 30,
        attention_module: Optional[nn.Module] = None,
        ffn_hidden_dim: int = 1536,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        use_bias: bool = False,
        tie_embeddings: bool = True,
        norm_strategy: str = "pre",
    ):
        super().__init__()

        if attention_module is None:
            raise ValueError("attention_module must be provided")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    attention_module=attention_module,
                    ffn_hidden_dim=ffn_hidden_dim,
                    dropout=dropout,
                    use_bias=use_bias,
                    norm_strategy=norm_strategy,
                )
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        self.final_norm = nn.RMSNorm(embed_dim)

        # Output projection (language modeling head)
        if tie_embeddings:
            # Share weights with input embeddings
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using scaled initialization."""
        # Token embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        # LM head (if not tied)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len)
            attention_mask (Optional[torch.Tensor]): Attention mask
            labels (Optional[torch.Tensor]): Target token IDs for computing loss

        Returns:
            dict: Dictionary containing:
                - logits (torch.Tensor): Output logits of shape (batch_size, seq_len, vocab_size)
                - loss (Optional[torch.Tensor]): Cross-entropy loss if labels provided
        """
        batch_size, seq_len = input_ids.shape

        # Get token embeddings
        x = self.token_embedding(input_ids)

        # Create causal attention mask if not provided
        if attention_mask is None:
            # Causal mask: True where j > i (future positions)
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
                diagonal=1,
            ).unsqueeze(0)  # (1, seq_len, seq_len)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        # Final normalization
        x = self.final_norm(x)

        # Project to vocabulary
        if self.tie_embeddings:
            # Use transposed token embeddings
            logits = F.linear(x, self.token_embedding.weight)
        else:
            if self.lm_head is not None:
                logits = self.lm_head(x)
            else:
                raise RuntimeError("lm_head is None but tie_embeddings is False")

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for loss computation
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # Standard padding index
            )

        return {
            "logits": logits,
            "loss": loss,
        }

    def count_parameters(self) -> int:
        """Count the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_smollm_with_attention(
    attention_module: nn.Module,
    vocab_size: int = 49152,
    embed_dim: int = 576,
    num_layers: int = 30,
    ffn_hidden_dim: int = 1536,
    max_seq_len: int = 2048,
    dropout: float = 0.0,
    use_bias: bool = False,
    tie_embeddings: bool = True,
    norm_strategy: str = "pre",
) -> SmolLM:
    """
    Factory function to create a SmolLM model with a specific attention mechanism.

    Args:
        attention_module (nn.Module): Attention module to use (MHA, GQA, MLA, etc.)
        vocab_size (int): Vocabulary size. Default: 49152
        embed_dim (int): Embedding dimension. Default: 576
        num_layers (int): Number of transformer layers. Default: 30
        ffn_hidden_dim (int): Hidden dimension for FFN. Default: 1536
        max_seq_len (int): Maximum sequence length. Default: 2048
        dropout (float): Dropout probability. Default: 0.0
        use_bias (bool): Whether to use bias. Default: False
        tie_embeddings (bool): Whether to tie embeddings. Default: True
        norm_strategy (str): Normalization strategy ('pre', 'post', 'sandwich'). Default: 'pre'

    Returns:
        SmolLM: Instantiated SmolLM model

    Example:
        >>> from nn_components.mha import MultiHeadAttention
        >>> attention = MultiHeadAttention(embed_dim=576, num_heads=8)
        >>> model = create_smollm_with_attention(attention)
        >>> print(f"Total parameters: {model.count_parameters():,}")
    """
    return SmolLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        attention_module=attention_module,
        ffn_hidden_dim=ffn_hidden_dim,
        max_seq_len=max_seq_len,
        dropout=dropout,
        use_bias=use_bias,
        tie_embeddings=tie_embeddings,
        norm_strategy=norm_strategy,
    )
