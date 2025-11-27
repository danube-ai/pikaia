import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn_modules.mga import MultiheadGeneticAttention


class BertModel(nn.Module):
    """
    The BERT model for sentence embeddings, based on the BERT architecture.
    It includes token embeddings, position embeddings, token type embeddings,
    an encoder stack, and mean pooling with L2 normalization.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 384,
        num_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 1536,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.1,
        use_genetic: bool = False,
    ):
        """
        Initialize the BERT model.

        Args:
            vocab_size (int):
                Size of the vocabulary.
            hidden_size (int):
                Dimensionality of the hidden states.
            num_layers (int):
                Number of encoder layers.
            num_attention_heads (int):
                Number of attention heads.
            intermediate_size (int):
                Dimensionality of the intermediate feed-forward layer.
            max_position_embeddings (int):
                Maximum sequence length for position embeddings.
            type_vocab_size (int):
                Number of token types.
            layer_norm_eps (float):
                Epsilon for layer normalization.
            dropout (float):
                Dropout probability.
            use_genetic (bool):
                Whether to use genetic attention in the encoder.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.emb_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.emb_dropout = nn.Dropout(dropout)

        self.encoder = BertEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            attention_dropout=dropout,
            use_genetic=use_genetic,
        )

        # final projection (sentence-transformers/all-BERT-L6-v2 keeps embedding
        # dim = hidden_size)
        # some variants add a projection; here we keep identity (you can add
        # nn.Linear if needed)
        # self.pooler = nn.Linear(hidden_size, hidden_size)
        # self.pooler_activation = nn.Tanh()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for the BERT model.

        Args:
            input_ids (torch.Tensor):
                Input token IDs of shape (batch, seq_len).
            attention_mask (torch.Tensor | None):
                Attention mask of shape (batch, seq_len).
                Values should be 1 for tokens to keep, 0 for padded tokens.
            token_type_ids (torch.Tensor | None):
                Token type IDs of shape (batch, seq_len).

        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - "token_embeddings": Tensor of shape (batch, seq_len, hidden_size).
                - "pooled_embedding": Tensor of shape (batch, hidden_size).
        """
        batch_size, seq_len = input_ids.size()
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        positions = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        tok_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(positions)
        type_emb = self.token_type_embeddings(token_type_ids)

        x = tok_emb + pos_emb + type_emb
        x = self.emb_layer_norm(x)
        x = self.emb_dropout(x)

        # Encoder (6 x transformer layers)
        token_embeddings = self.encoder(x, attn_mask=attention_mask)

        # Mean pooling over valid tokens (attention_mask == 1)
        mask = attention_mask.unsqueeze(-1).to(
            token_embeddings.dtype
        )  # (batch, seq_len, 1)
        summed = torch.sum(token_embeddings * mask, dim=1)  # (batch, hidden)
        lengths = torch.clamp(mask.sum(dim=1), min=1e-9)  # (batch, 1)
        mean_pooled = summed / lengths

        # L2-normalize (sentence-transformers uses normalized embeddings)
        normalized = F.normalize(mean_pooled, p=2, dim=1)

        return {
            "token_embeddings": token_embeddings,
            "pooled_embedding": normalized,
        }


class BertEncoder(nn.Module):
    """
    The encoder component of the BERT model, consisting of a stack of
    BertEncoderLayer modules.
    """

    def __init__(
        self,
        num_layers: int = 6,
        hidden_size: int = 384,
        num_attention_heads: int = 12,
        intermediate_size: int = 1536,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_genetic: bool = False,
    ):
        """
        Initialize the BERT encoder.

        Args:
            num_layers (int):
                Number of encoder layers.
            hidden_size (int):
                Dimensionality of the hidden states.
            num_attention_heads (int):
                Number of attention heads per layer.
            intermediate_size (int):
                Dimensionality of the intermediate feed-forward layer.
            dropout (float):
                Dropout probability.
            attention_dropout (float):
                Dropout probability for attention.
            use_genetic (bool):
                Whether to use genetic attention in encoder layers.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BertEncoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    use_genetic=use_genetic,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass for the BERT encoder.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch, seq_len, hidden_size).
            attn_mask (torch.Tensor | None):
                Attention mask of shape (batch, seq_len) or (batch, seq_len, seq_len).
                Values should be 1 for tokens to keep, 0 for padded tokens.

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, hidden_size).
        """
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x


class BertEncoderLayer(nn.Module):
    """
    A single encoder layer for the BERT model, consisting of multi-head self-attention
    and a feed-forward network with residual connections and layer normalization.
    """

    def __init__(
        self,
        hidden_size: int = 384,
        num_attention_heads: int = 12,
        intermediate_size: int = 1536,  # usually 4 * hidden_size
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_genetic: bool = False,
    ):
        """
        Initialize the BERT encoder layer.

        Args:
            hidden_size (int):
                Dimensionality of the hidden states.
            num_attention_heads (int):
                Number of attention heads.
            intermediate_size (int):
                Dimensionality of the intermediate feed-forward layer.
            dropout (float):
                Dropout probability for the feed-forward network.
            attention_dropout (float):
                Dropout probability for attention.
            use_genetic (bool):
                Whether to use genetic attention instead of standard multi-head attention.
        """
        super().__init__()
        if use_genetic:
            self.self_attn = MultiheadGeneticAttention(
                embed_dim=hidden_size,
                n_heads=num_attention_heads,
                dropout=attention_dropout,
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                batch_first=True,
                dropout=attention_dropout,
            )
        self.use_genetic = use_genetic
        self.attn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass for the BERT encoder layer.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch, seq_len, hidden_size).
            attn_mask (torch.Tensor | None):
                Attention mask of shape (batch, seq_len) or (batch, seq_len, seq_len).
                Values should be 1 for tokens to keep, 0 for padded tokens.

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, hidden_size).
        """
        if attn_mask is not None:
            # assume attn_mask is 1 for tokens to keep, 0 for pad
            # key_padding_mask expects True for positions that are masked (to be
            # ignored)
            if attn_mask.dim() == 2:
                key_padding_mask = ~attn_mask.to(torch.bool)  # (batch, seq_len)
            else:
                # if it is (batch, seq_len, seq_len) fallback to None
                key_padding_mask = None
        else:
            key_padding_mask = None

        # Self-attention block (residual)
        if self.use_genetic:
            # MGA expects attn_mask as (batch, seq_len) with True for valid tokens
            mga_attn_mask = attn_mask.to(torch.bool) if attn_mask is not None else None
            attn_out = self.self_attn(x, attn_mask=mga_attn_mask, disable_genetic=False)
        else:
            attn_out, _ = self.self_attn(
                x, x, x, key_padding_mask=key_padding_mask, need_weights=False
            )
        x = x + self.dropout(attn_out)
        x = self.attn_layer_norm(x)

        # FFN block (residual)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.ffn_layer_norm(x)

        return x
