"""
Genetic Layer Module
"""

from typing import Optional

import torch
import torch.nn as nn


class GeneticLayer(nn.Module):
    """
    A PyTorch nn.Module implementing a genetic-inspired layer for feed-forward operations.

    This layer computes an internal population matrix from the input, applies sigmoid activation,
    and then uses a fixed-point formula (based on dominant gene and balanced organism strategies)
    to compute organism fitness values, which serve as the output.

    Parameters:
        input_shape (int):
            Number of features in the input.
        orgs_shape (int):
            Number of latent organisms.
        genes_shape (int):
            Number of latent genes.
        strategy (str):
            The strategy for computing fitness. Currently only 'fixed_point' is
            supported.
        hidden_dim (int):
            Dimension to project the input to before the genetic computation.
            Defaults to input_shape if not provided.
        output_shape (int):
            Dimension to project the output to after the genetic computation.
            Defaults to orgs_shape if not provided.
        activation_fn (nn.Module):
            Activation function to use in projections. Defaults to SiLU.
        dropout_rate (float):
            Dropout rate to apply after activations. Defaults to 0.1.

    Input:
        x (torch.Tensor):
            Input tensor of shape (..., input_shape), where the last dimension is features.

    Output:
        torch.Tensor:
            Output tensor of shape (..., output_shape) if output_shape is provided,
            otherwise (..., orgs_shape) containing organism fitness values.
    """

    def __init__(
        self,
        input_shape: int,
        hidden_dim: int = 512,
        orgs_shape: int = 32,
        genes_shape: int = 8,
        strategy: str = "fixed_org_balanced_gene_dominant",
        output_shape: Optional[int] = None,
        activation_fn: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.orgs_shape = orgs_shape
        self.genes_shape = genes_shape
        self.strategy = strategy
        self.hidden_dim = hidden_dim
        self.output_shape = output_shape if output_shape is not None else orgs_shape
        self.activation_fn = activation_fn if activation_fn is not None else nn.SiLU()
        self.dropout_rate = dropout_rate

        # Create sub-modules
        self.input_projection = InputProjection(
            input_shape, hidden_dim, self.activation_fn, dropout_rate
        )
        self.genetic_projection = GeneticProjection(
            hidden_dim, orgs_shape, genes_shape, self.activation_fn, dropout_rate
        )
        self.strategy_module = StrategyModule(strategy)
        self.output_projection = OutputProjection(
            orgs_shape, self.output_shape, self.activation_fn, dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GeneticLayer.

        Args:
            x (torch.Tensor):
                Input tensor of shape (..., input_shape), where the last dimension is features.

        Returns:
            torch.Tensor:
                Output tensor of shape (..., output_shape) if output_shape is provided,
                otherwise (..., orgs_shape).
        """
        # 1. Validate input and extract shape information
        original_shape = x.shape
        if len(original_shape) < 2:
            raise ValueError(
                "Input tensor must have at least 2 dimensions (batch and features)."
            )
        batch_size = original_shape[0]
        middle_shape = original_shape[1:-1]
        flattened_middle = (
            int(torch.prod(torch.tensor(middle_shape))) if middle_shape else 1
        )
        input_shape = original_shape[-1]

        # 2. Reshape input to 3D
        # Output shape: (batch_size, flattened_middle, input_shape)
        x = x.view(batch_size, flattened_middle, input_shape)

        # 3. Apply input projection:
        # Output shape: (batch_size, flattened_middle, hidden_dim)
        x = self.input_projection(x)

        # 4. Apply genetic projection:
        # Output shape: (batch_size, input_length, orgs_shape, genes_shape)
        population_matrix = self.genetic_projection(
            x, batch_size, flattened_middle, self.orgs_shape, self.genes_shape
        )

        # 5. Apply genetic strategy computation
        # Output shape: (batch_size, input_length, orgs_shape)
        org_fitness = self.strategy_module(population_matrix)

        # 6. Apply output projection: LayerNorm → Linear → SiLU → Dropout
        # Output shape: (batch_size, flattened_middle, output_shape)
        org_fitness = self.output_projection(org_fitness)

        # 7. Reshape output back to original shape format
        # Output shape: (batch_size,) + middle_shape + (output_shape,)
        output_shape_tuple = (batch_size,) + middle_shape + (self.output_shape,)
        org_fitness = org_fitness.view(output_shape_tuple)

        return org_fitness


class InputProjection(nn.Module):
    """
    Input projection module that transforms input features to hidden dimensions.

    This module applies layer normalization, linear transformation, activation,
    and dropout to project the input features into a higher-dimensional hidden space
    suitable for genetic computations.

    Parameters:
        input_shape (int):
            Number of input features.
        hidden_dim (int):
            Number of hidden dimensions to project to.
        activation_fn (nn.Module):
            Activation function to apply after linear transformation.
        dropout_rate (float):
            Dropout probability. If 0, no dropout is applied.

    Attributes:
        layer_norm (nn.LayerNorm):
            Layer normalization applied to input features.
        linear (nn.Linear):
            Linear layer for input projection.
        activation (nn.Module):
            Activation function module.
        dropout (nn.Module):
            Dropout module (nn.Dropout or nn.Identity).
    """

    def __init__(
        self,
        input_shape: int,
        hidden_dim: int,
        activation_fn: nn.Module,
        dropout_rate: float,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_shape)
        self.linear = nn.Linear(input_shape, hidden_dim)
        self.activation = activation_fn
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the input projection.

        Args:
            x (torch.Tensor):
                Input tensor of shape (..., input_shape).

        Returns:
            torch.Tensor:
                Output tensor of shape (..., hidden_dim) after layer normalization,
                linear transformation, activation, and dropout.
        """
        x = self.layer_norm(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class GeneticProjection(nn.Module):
    """
    Genetic weight projection module that transforms hidden features to genetic population matrix.

    This module applies layer normalization, linear transformation, activation,
    and dropout, then reshapes and applies sigmoid activation to create the population
    matrix used in genetic computations.

    Parameters:
        hidden_dim (int):
            Number of hidden dimensions from input projection.
        orgs_shape (int):
            Number of latent organisms in the population.
        genes_shape (int):
            Number of latent genes per organism.
        activation_fn (nn.Module):
            Activation function to apply after linear transformation.
        dropout_rate (float):
            Dropout probability. If 0, no dropout is applied.

    Attributes:
        layer_norm (nn.LayerNorm):
            Layer normalization applied to hidden features.
        linear (nn.Linear):
            Linear layer that outputs orgs_shape * genes_shape features.
        activation (nn.Module):
            Activation function module.
        dropout (nn.Module):
            Dropout module (nn.Dropout or nn.Identity).
    """

    def __init__(
        self,
        hidden_dim: int,
        orgs_shape: int,
        genes_shape: int,
        activation_fn: nn.Module,
        dropout_rate: float,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, orgs_shape * genes_shape)
        self.activation = activation_fn
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        batch_size: int,
        input_length: int,
        orgs_shape: int,
        genes_shape: int,
    ) -> torch.Tensor:
        """
        Forward pass through the genetic projection.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, input_length, hidden_dim).
            batch_size (int):
                Batch size of the input.
            input_length (int):
                Length of the input sequence (flattened middle dimensions).
            orgs_shape (int):
                Number of organisms.
            genes_shape (int):
                Number of genes per organism.

        Returns:
            torch.Tensor:
                Population matrix of shape (batch_size, input_length, orgs_shape, genes_shape)
                with values in [0, 1] after sigmoid activation.
        """
        x = self.layer_norm(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        # Reshape and apply sigmoid
        population_matrix = x.view(batch_size, input_length, orgs_shape, genes_shape)
        population_matrix = torch.sigmoid(population_matrix)
        return population_matrix


class StrategyModule(nn.Module):
    """
    Strategy computation module that implements genetic fitness calculation strategies.

    This module encapsulates different strategies for computing organism fitness from
    the population matrix. Currently supports the fixed-point strategy based on
    dominant gene and balanced organism principles.

    Parameters:
        strategy (str):
            The strategy to use for fitness computation. Currently only
            'fixed_org_balanced_gene_dominant' is supported.

    Attributes:
        strategy (str):
            The selected strategy name.
        _run_strategy (callable):
            Internal method that implements the selected strategy.

    Raises:
        ValueError:
            If an unsupported strategy is provided.
    """

    def __init__(self, strategy: str):
        super().__init__()
        self.strategy = strategy
        match self.strategy:
            case "fixed_org_balanced_gene_dominant":
                self._run_strategy = self._fixed_org_balanced_gene_dominant_strategy
            case _:
                raise ValueError(
                    f"Unsupported strategy: {self.strategy}. "
                    "Only 'fixed_org_balanced_gene_dominant' is currently supported."
                )

    def forward(self, population_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute organism fitness using the selected strategy.

        Args:
            population_matrix (torch.Tensor):
                Population matrix of shape (batch_size, input_length, orgs_shape, genes_shape)
                with values in [0, 1].

        Returns:
            torch.Tensor:
                Organism fitness values of shape (batch_size, input_length, orgs_shape).
        """
        return self._run_strategy(population_matrix)

    def _fixed_org_balanced_gene_dominant_strategy(
        self, population_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes organism fitness using the fixed-point formula based on
            dominant gene and balanced organism strategies.

        Args:
            population_matrix (torch.Tensor):
                Population matrix of shape
                (batch_size, input_length, orgs_shape, genes_shape).

        Returns:
            torch.Tensor:
                Organism fitness of shape (batch_size, input_length, orgs_shape).
        """
        # (batch_size, input_length, genes_shape)
        gene_means = torch.mean(population_matrix, dim=2)

        # (batch_size, input_length, genes_shape)
        denom = gene_means + 0.5

        # (batch_size, input_length, 1)
        sum_inv_denom = torch.sum(1 / denom, dim=2, keepdim=True)

        # (batch_size, input_length, genes_shape)
        gene_fitness = 1 / (denom * sum_inv_denom)

        # (batch_size, input_length, orgs_shape)
        org_fitness = torch.matmul(
            population_matrix, gene_fitness.unsqueeze(-1)
        ).squeeze(-1)

        return org_fitness


class OutputProjection(nn.Module):
    """
    Output projection module that transforms organism fitness to final output dimensions.

    This module applies layer normalization, linear transformation, activation,
    and dropout to project the organism fitness values to the desired output shape.

    Parameters:
        orgs_shape (int):
            Number of organisms (input dimension).
        output_shape (int):
            Number of output features.
        activation_fn (nn.Module):
            Activation function to apply after linear transformation.
        dropout_rate (float):
            Dropout probability. If 0, no dropout is applied.

    Attributes:
        layer_norm (nn.LayerNorm):
            Layer normalization applied to organism fitness values.
        linear (nn.Linear):
            Linear layer for output projection.
        activation (nn.Module):
            Activation function module.
        dropout (nn.Module):
            Dropout module (nn.Dropout or nn.Identity).
    """

    def __init__(
        self,
        orgs_shape: int,
        output_shape: int,
        activation_fn: nn.Module,
        dropout_rate: float,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(orgs_shape)
        self.linear = nn.Linear(orgs_shape, output_shape)
        self.activation = activation_fn
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the output projection.

        Args:
            x (torch.Tensor):
                Input tensor of shape (..., orgs_shape) containing organism fitness values.

        Returns:
            torch.Tensor:
                Output tensor of shape (..., output_shape) after layer normalization,
                linear transformation, activation, and dropout.
        """
        x = self.layer_norm(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
