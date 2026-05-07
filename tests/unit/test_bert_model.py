"""Tests for pikaia.models.backbones.bert and related model stubs."""

import numpy as np
import pytest
import torch

from pikaia.data.population import PikaiaPopulation
from pikaia.models.backbones.bert import BertEncoder, BertEncoderLayer, BertModel
from pikaia.models.danube_model import DanubeModel
from pikaia.models.pikaia_text_embedder import PikaiaTextEmbedder


# ---------------------------------------------------------------------------
# BertModel
# ---------------------------------------------------------------------------


class TestBertModel:
    """Tests for the BertModel forward pass."""

    @pytest.fixture
    def small_model(self):
        return BertModel(
            vocab_size=100,
            hidden_size=32,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=64,
        )

    def test_forward_output_keys(self, small_model):
        input_ids = torch.randint(0, 100, (2, 16))
        out = small_model(input_ids)
        assert "token_embeddings" in out
        assert "pooled_embedding" in out

    def test_forward_token_embeddings_shape(self, small_model):
        input_ids = torch.randint(0, 100, (2, 16))
        out = small_model(input_ids)
        assert out["token_embeddings"].shape == (2, 16, 32)

    def test_forward_pooled_embedding_shape(self, small_model):
        input_ids = torch.randint(0, 100, (2, 16))
        out = small_model(input_ids)
        assert out["pooled_embedding"].shape == (2, 32)

    def test_forward_pooled_embedding_normalized(self, small_model):
        """Pooled embeddings should be L2-normalised (norm ≈ 1)."""
        input_ids = torch.randint(0, 100, (3, 10))
        out = small_model(input_ids)
        norms = out["pooled_embedding"].norm(dim=1)
        torch.testing.assert_close(norms, torch.ones(3), atol=1e-5, rtol=0)

    def test_forward_with_attention_mask(self, small_model):
        input_ids = torch.randint(0, 100, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.long)
        mask[0, 12:] = 0  # pad last 4 tokens in first sequence
        out = small_model(input_ids, attention_mask=mask)
        assert out["pooled_embedding"].shape == (2, 32)

    def test_forward_with_token_type_ids(self, small_model):
        input_ids = torch.randint(0, 100, (2, 16))
        token_type_ids = torch.zeros(2, 16, dtype=torch.long)
        out = small_model(input_ids, token_type_ids=token_type_ids)
        assert out["pooled_embedding"].shape == (2, 32)

    def test_forward_batch_size_one(self, small_model):
        input_ids = torch.randint(0, 100, (1, 8))
        out = small_model(input_ids)
        assert out["pooled_embedding"].shape == (1, 32)

    def test_use_genetic_true(self):
        """BertModel with use_genetic=True should initialise without error."""
        model = BertModel(
            vocab_size=50,
            hidden_size=32,
            num_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            use_genetic=True,
        )
        input_ids = torch.randint(0, 50, (2, 8))
        out = model(input_ids)
        assert out["pooled_embedding"].shape == (2, 32)


# ---------------------------------------------------------------------------
# BertEncoder
# ---------------------------------------------------------------------------


class TestBertEncoder:
    """Tests for the BertEncoder forward pass."""

    @pytest.fixture
    def encoder(self):
        return BertEncoder(
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            intermediate_size=64,
        )

    def test_forward_shape(self, encoder):
        x = torch.randn(2, 10, 32)
        out = encoder(x)
        assert out.shape == (2, 10, 32)

    def test_forward_with_mask(self, encoder):
        x = torch.randn(2, 10, 32)
        mask = torch.ones(2, 10, dtype=torch.long)
        out = encoder(x, attn_mask=mask)
        assert out.shape == (2, 10, 32)


# ---------------------------------------------------------------------------
# BertEncoderLayer
# ---------------------------------------------------------------------------


class TestBertEncoderLayer:
    """Tests for BertEncoderLayer."""

    @pytest.fixture
    def layer(self):
        return BertEncoderLayer(
            hidden_size=32,
            num_attention_heads=4,
            intermediate_size=64,
        )

    def test_forward_no_mask(self, layer):
        x = torch.randn(2, 10, 32)
        out = layer(x)
        assert out.shape == (2, 10, 32)

    def test_forward_2d_mask(self, layer):
        x = torch.randn(2, 10, 32)
        mask = torch.ones(2, 10, dtype=torch.long)
        out = layer(x, attn_mask=mask)
        assert out.shape == (2, 10, 32)

    def test_forward_3d_mask_fallback(self, layer):
        """3D attention mask falls back to no key_padding_mask."""
        x = torch.randn(2, 10, 32)
        mask = torch.ones(2, 10, 10, dtype=torch.long)
        out = layer(x, attn_mask=mask)
        assert out.shape == (2, 10, 32)

    def test_genetic_layer(self):
        layer = BertEncoderLayer(
            hidden_size=32,
            num_attention_heads=4,
            intermediate_size=64,
            use_genetic=True,
        )
        x = torch.randn(2, 10, 32)
        out = layer(x)
        assert out.shape == (2, 10, 32)


# ---------------------------------------------------------------------------
# PikaiaTextEmbedder
# ---------------------------------------------------------------------------


class TestPikaiaTextEmbedder:
    """Tests for PikaiaTextEmbedder stub."""

    def test_init_default_model_name(self):
        embedder = PikaiaTextEmbedder()
        assert embedder.model_name == "default-model"

    def test_init_custom_model_name(self):
        embedder = PikaiaTextEmbedder(model_name="bert-base")
        assert embedder.model_name == "bert-base"

    def test_embed_raises_not_implemented(self):
        embedder = PikaiaTextEmbedder()
        with pytest.raises(NotImplementedError, match="not implemented"):
            embedder.embed("hello world")


# ---------------------------------------------------------------------------
# DanubeModel
# ---------------------------------------------------------------------------


class TestDanubeModel:
    """Tests for DanubeModel stub."""

    @pytest.fixture
    def model(self):
        rng = np.random.default_rng(0)
        pop = PikaiaPopulation(rng.random((3, 4)))
        return DanubeModel(population=pop)

    def test_fit_raises_not_implemented(self, model):
        with pytest.raises(NotImplementedError, match="fit"):
            model.fit()

    def test_predict_raises_not_implemented(self, model):
        rng = np.random.default_rng(1)
        pop = PikaiaPopulation(rng.random((2, 4)))
        with pytest.raises(NotImplementedError, match="predict"):
            model.predict(pop)
