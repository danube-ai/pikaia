"""Tests for pikaia.models.nn_modules.genetic_layer."""

import pytest
import torch
import torch.nn as nn

from pikaia.models.nn_modules.genetic_layer import (
    GeneticLayer,
    GeneticProjection,
    InputProjection,
    OutputProjection,
    StrategyModule,
)


class TestGeneticLayer:
    def test_forward_2d_input(self):
        layer = GeneticLayer(input_shape=16, hidden_dim=32, orgs_shape=8, genes_shape=4)
        x = torch.randn(2, 16)
        out = layer(x)
        assert out.shape == (2, 8)

    def test_forward_3d_input(self):
        layer = GeneticLayer(input_shape=16, hidden_dim=32, orgs_shape=8, genes_shape=4)
        x = torch.randn(2, 5, 16)
        out = layer(x)
        assert out.shape == (2, 5, 8)

    def test_forward_with_output_shape(self):
        layer = GeneticLayer(
            input_shape=16, hidden_dim=32, orgs_shape=8, genes_shape=4, output_shape=12
        )
        x = torch.randn(2, 16)
        out = layer(x)
        assert out.shape == (2, 12)

    def test_forward_1d_raises(self):
        layer = GeneticLayer(input_shape=16, hidden_dim=32, orgs_shape=8, genes_shape=4)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            layer(torch.randn(16))

    def test_custom_activation(self):
        layer = GeneticLayer(
            input_shape=8, hidden_dim=16, orgs_shape=4, genes_shape=2,
            activation_fn=nn.ReLU(),
        )
        x = torch.randn(2, 8)
        out = layer(x)
        assert out.shape == (2, 4)

    def test_zero_dropout(self):
        layer = GeneticLayer(
            input_shape=8, hidden_dim=16, orgs_shape=4, genes_shape=2, dropout_rate=0.0
        )
        x = torch.randn(2, 8)
        out = layer(x)
        assert out.shape == (2, 4)


class TestInputProjection:
    def test_forward(self):
        proj = InputProjection(input_shape=16, hidden_dim=32, activation_fn=nn.SiLU(), dropout_rate=0.1)
        x = torch.randn(2, 5, 16)
        out = proj(x)
        assert out.shape == (2, 5, 32)

    def test_zero_dropout_uses_identity(self):
        proj = InputProjection(input_shape=8, hidden_dim=16, activation_fn=nn.SiLU(), dropout_rate=0.0)
        assert isinstance(proj.dropout, nn.Identity)
        x = torch.randn(2, 5, 8)
        out = proj(x)
        assert out.shape == (2, 5, 16)


class TestGeneticProjection:
    def test_forward_shape(self):
        proj = GeneticProjection(
            hidden_dim=32, orgs_shape=8, genes_shape=4,
            activation_fn=nn.SiLU(), dropout_rate=0.1,
        )
        x = torch.randn(2, 5, 32)
        out = proj(x, batch_size=2, input_length=5, orgs_shape=8, genes_shape=4)
        assert out.shape == (2, 5, 8, 4)

    def test_output_values_in_unit_interval(self):
        proj = GeneticProjection(
            hidden_dim=16, orgs_shape=4, genes_shape=2,
            activation_fn=nn.SiLU(), dropout_rate=0.0,
        )
        x = torch.randn(2, 3, 16)
        out = proj(x, batch_size=2, input_length=3, orgs_shape=4, genes_shape=2)
        assert out.min() >= 0.0 and out.max() <= 1.0


class TestStrategyModule:
    def test_valid_strategy(self):
        module = StrategyModule("fixed_org_balanced_gene_dominant")
        pop = torch.rand(2, 5, 8, 4)
        out = module(pop)
        assert out.shape == (2, 5, 8)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unsupported strategy"):
            StrategyModule("nonexistent_strategy")


class TestOutputProjection:
    def test_forward(self):
        proj = OutputProjection(orgs_shape=8, output_shape=12, activation_fn=nn.SiLU(), dropout_rate=0.1)
        x = torch.randn(2, 5, 8)
        out = proj(x)
        assert out.shape == (2, 5, 12)

    def test_zero_dropout_uses_identity(self):
        proj = OutputProjection(orgs_shape=8, output_shape=4, activation_fn=nn.SiLU(), dropout_rate=0.0)
        assert isinstance(proj.dropout, nn.Identity)
