import torch

from pikaia.models.nn_modules import MultiheadGeneticAttention


class TestMultiheadGeneticAttention:
    def test_standard_multihead_attention(self):
        """Test standard multihead attention (no GQA, no MLA)."""
        attn = MultiheadGeneticAttention(embed_dim=64, n_heads=8)
        x = torch.randn(2, 10, 64)
        y = attn(x)
        assert y.shape == (2, 10, 64)
        assert torch.isfinite(y).all()

    def test_gqa(self):
        """Test Grouped Query Attention with fewer KV heads."""
        attn = MultiheadGeneticAttention(embed_dim=64, n_heads=8, n_kv_heads=4)
        x = torch.randn(2, 10, 64)
        y = attn(x)
        assert y.shape == (2, 10, 64)
        assert torch.isfinite(y).all()

    def test_mla(self):
        """Test Multi-Head Latent Attention with input projection."""
        attn = MultiheadGeneticAttention(embed_dim=64, n_heads=8, in_proj_dim=32)
        x = torch.randn(2, 10, 64)
        y = attn(x)
        assert y.shape == (2, 10, 64)
        assert torch.isfinite(y).all()

    def test_gqa_and_mla(self):
        """Test both GQA and MLA together."""
        attn = MultiheadGeneticAttention(
            embed_dim=64, n_heads=8, n_kv_heads=4, in_proj_dim=32
        )
        x = torch.randn(2, 10, 64)
        y = attn(x)
        assert y.shape == (2, 10, 64)
        assert torch.isfinite(y).all()

    def test_attention_mask(self):
        """Test attention mask functionality."""
        # Create a padding mask: (batch_size, seq_len) where True = valid, False = padding
        # For this test, make the last 2 tokens of each sequence padding
        seq_len = 10
        mask = torch.ones(2, seq_len, dtype=torch.bool)  # (batch_size, seq_len)
        mask[:, -2:] = False  # Last 2 positions are padding

        attn = MultiheadGeneticAttention(embed_dim=64, n_heads=8)
        x = torch.randn(2, 10, 64)
        y = attn(x, attn_mask=mask)
        assert y.shape == (2, 10, 64)
        assert torch.isfinite(y).all()

    def test_invalid_head_dim(self):
        """Test that invalid head dimension raises ValueError."""
        try:
            MultiheadGeneticAttention(embed_dim=63, n_heads=8)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "must be divisible" in str(e)
