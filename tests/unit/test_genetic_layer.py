import torch

from pikaia.models.nn_modules.genetic_layer import GeneticLayer


class TestGeneticLayer:
    def test_forward_1d_input_raises_error(self):
        """Test that 1D input raises ValueError."""
        layer = GeneticLayer(64, 32, 16)
        x = torch.randn(64)  # 1D
        try:
            layer(x)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "at least 2 dimensions" in str(e)

    def test_forward_2d_input(self):
        """Test GeneticLayer with 2D input (batch_size, features)."""
        layer = GeneticLayer(64, 32, 16)
        x = torch.randn(10, 64)
        y = layer(x)
        assert y.shape == (10, 16)
        assert torch.isfinite(y).all()

    def test_forward_3d_input(self):
        """Test GeneticLayer with 3D input (batch_size, seq_len, features)."""
        layer = GeneticLayer(64, 32, 16)
        x = torch.randn(10, 5, 64)
        y = layer(x)
        assert y.shape == (10, 5, 16)
        assert torch.isfinite(y).all()

    def test_forward_4d_input(self):
        """Test GeneticLayer with 4D input (batch, height, width, features)."""
        layer = GeneticLayer(64, 32, 16)
        x = torch.randn(2, 3, 4, 64)
        y = layer(x)
        assert y.shape == (2, 3, 4, 16)
        assert torch.isfinite(y).all()

    def test_forward_5d_input(self):
        """Test GeneticLayer with 5D input."""
        layer = GeneticLayer(64, 32, 16)
        x = torch.randn(2, 3, 4, 5, 64)
        y = layer(x)
        assert y.shape == (2, 3, 4, 5, 16)
        assert torch.isfinite(y).all()

    def test_single_batch(self):
        """Test with single batch."""
        layer = GeneticLayer(64, 32, 16)
        x = torch.randn(1, 64)
        y = layer(x)
        assert y.shape == (1, 16)

    def test_single_feature(self):
        """Test with single feature."""
        layer = GeneticLayer(1, 32, 16)
        x = torch.randn(10, 1)
        y = layer(x)
        assert y.shape == (10, 16)

    def test_large_middle_dims(self):
        """Test with large flattened middle dimensions."""
        layer = GeneticLayer(64, 32, 16)
        x = torch.randn(2, 10, 20, 64)  # middle: 10*20=200
        y = layer(x)
        assert y.shape == (2, 10, 20, 16)

    def test_output_values_in_range(self):
        """Test that output values are reasonable."""
        layer = GeneticLayer(64, 32, 16)
        x = torch.randn(10, 64)
        y = layer(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()
        # Check that values are not all zero (unlikely but possible)
        assert y.sum() != 0

    def test_different_orgs_shapes(self):
        """Test with different orgs_shape values."""
        for orgs in [1, 16, 32, 64, 128]:
            layer = GeneticLayer(64, 512, orgs, 16)
            x = torch.randn(5, 64)
            y = layer(x)
            assert y.shape == (5, orgs)

    def test_different_genes_shapes(self):
        """Test with different genes_shape values."""
        for genes in [1, 8, 16, 32]:
            layer = GeneticLayer(64, 32, 32, genes)
            x = torch.randn(5, 64)
            y = layer(x)
            assert y.shape == (5, 32)

    def test_deterministic_output(self):
        """Test that output is deterministic for same input."""
        layer = GeneticLayer(64, 32, 16, dropout_rate=0.0)
        x = torch.randn(10, 64)
        y1 = layer(x)
        y2 = layer(x)
        assert torch.allclose(y1, y2)

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = GeneticLayer(64, 32, 16)
        x = torch.randn(10, 64, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_shape_equivalence_with_nn_linear(self):
        """Test that shape handling matches nn.Linear."""
        import torch.nn as nn

        linear = nn.Linear(64, 32)
        genetic = GeneticLayer(64, 32, 32, 16)

        # 2D
        x2d = torch.randn(10, 64)
        y_linear = linear(x2d)
        y_genetic = genetic(x2d)
        assert y_linear.shape == y_genetic.shape

        # 3D
        x3d = torch.randn(10, 5, 64)
        y_linear = linear(x3d)
        y_genetic = genetic(x3d)
        assert y_linear.shape == y_genetic.shape

        # 4D
        x4d = torch.randn(2, 3, 4, 64)
        y_linear = linear(x4d)
        y_genetic = genetic(x4d)
        assert y_linear.shape == y_genetic.shape

    def test_forward_with_hidden_dim(self):
        """Test GeneticLayer with hidden_dim projection."""
        layer = GeneticLayer(64, 32, 16, 32)
        x = torch.randn(10, 64)
        y = layer(x)
        assert y.shape == (10, 16)
        assert torch.isfinite(y).all()

    def test_forward_with_output_shape(self):
        """Test GeneticLayer with output_shape projection."""
        layer = GeneticLayer(64, 32, 16, output_shape=10)
        x = torch.randn(10, 64)
        y = layer(x)
        assert y.shape == (10, 10)
        assert torch.isfinite(y).all()

    def test_forward_with_both_projections(self):
        """Test GeneticLayer with both hidden_dim and output_shape."""
        layer = GeneticLayer(64, 32, 16, 32, output_shape=10)
        x = torch.randn(10, 64)
        y = layer(x)
        assert y.shape == (10, 10)
        assert torch.isfinite(y).all()
