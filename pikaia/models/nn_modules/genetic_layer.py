"""
Genetic Layer Module
"""

import torch
import torch.nn as nn


class GeneticLayer(nn.Module):
    """
    A PyTorch nn.Module implementing a genetic-inspired feed-forward layer.

    Projects the input through a hidden space, constructs a latent population
    matrix (organisms × genes) via sigmoid activation, applies a fixed-point
    genetic fitness formula (dominant gene + balanced organism strategies), and
    projects the resulting organism fitness scores to the desired output shape.
    """

    def __init__(
        self,
        input_shape: int,
        hidden_dim: int = 512,
        orgs_shape: int = 32,
        genes_shape: int = 8,
        strategy: str = "fixed_org_balanced_gene_dominant",
        output_shape: int | None = None,
        activation_fn: nn.Module | None = None,
        dropout_rate: float = 0.1,
    ):
        """Initialise GeneticLayer.

        Args:
            input_shape: Number of features in the last dimension of the input.
            hidden_dim: Intermediate dimension of the input projection. Defaults
                to ``512``.
            orgs_shape: Number of latent organisms in the population matrix.
                Defaults to ``32``.
            genes_shape: Number of latent genes per organism. Defaults to ``8``.
            strategy: Fitness computation strategy.  Currently only
                ``"fixed_org_balanced_gene_dominant"`` is supported.
            output_shape: Output feature dimension after the organism fitness
                projection.  Defaults to ``orgs_shape``.
            activation_fn: Activation function used in all projection modules.
                Defaults to :class:`torch.nn.SiLU`.
            dropout_rate: Dropout probability applied after activations. Defaults
                to ``0.1``.
        """
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
                Input tensor of shape (..., input_shape), where the last dimension
                is features.

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
    Input projection: LayerNorm → Linear → activation → Dropout.

    Projects input features into a higher-dimensional hidden space suitable
    for the genetic population computation.
    """

    def __init__(
        self,
        input_shape: int,
        hidden_dim: int,
        activation_fn: nn.Module,
        dropout_rate: float,
    ):
        """Initialise InputProjection.

        Args:
            input_shape: Number of input features.
            hidden_dim: Number of hidden dimensions to project to.
            activation_fn: Activation function applied after the linear layer.
            dropout_rate: Dropout probability. ``0`` disables dropout.
        """
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
    Genetic projection: LayerNorm → Linear → activation → Dropout → sigmoid.

    Transforms hidden features into a population matrix of shape
    ``(batch, length, orgs_shape, genes_shape)`` with values in ``[0, 1]``.
    """

    def __init__(
        self,
        hidden_dim: int,
        orgs_shape: int,
        genes_shape: int,
        activation_fn: nn.Module,
        dropout_rate: float,
    ):
        """Initialise GeneticProjection.

        Args:
            hidden_dim: Number of hidden dimensions from the input projection.
            orgs_shape: Number of latent organisms in the population.
            genes_shape: Number of latent genes per organism.
            activation_fn: Activation function applied after the linear layer.
            dropout_rate: Dropout probability. ``0`` disables dropout.
        """
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
                Population matrix of shape (batch_size, input_length, orgs_shape,
                genes_shape) with values in [0, 1] after sigmoid activation.
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
    Genetic fitness computation strategy encapsulation.

    Wraps a named strategy function for computing organism fitness from a
    population matrix.  Currently only ``"fixed_org_balanced_gene_dominant"``
    is supported.

    Raises:
        ValueError: If an unsupported strategy name is provided.
    """

    def __init__(self, strategy: str):
        """Initialise StrategyModule.

        Args:
            strategy: Name of the fitness computation strategy.  Currently only
                ``"fixed_org_balanced_gene_dominant"`` is supported.

        Raises:
            ValueError: If ``strategy`` is not a recognised name.
        """
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
                Population matrix of shape (batch_size, input_length, orgs_shape,
                genes_shape) with values in [0, 1].

        Returns:
            torch.Tensor:
                Organism fitness values of shape (batch_size, input_length, orgs_shape).
        """
        return self._run_strategy(population_matrix)

    def _fixed_org_balanced_gene_dominant_strategy(
        self, population_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute organism fitness via the dominant-gene fixed-point formula.

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
    Output projection: LayerNorm → Linear → activation → Dropout.

    Projects organism fitness scores to the final output dimensionality.
    """

    def __init__(
        self,
        orgs_shape: int,
        output_shape: int,
        activation_fn: nn.Module,
        dropout_rate: float,
    ):
        """Initialise OutputProjection.

        Args:
            orgs_shape: Number of organisms (input dimension).
            output_shape: Number of output features.
            activation_fn: Activation function applied after the linear layer.
            dropout_rate: Dropout probability. ``0`` disables dropout.
        """
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
                Input tensor of shape (..., orgs_shape) containing organism
                fitness values.

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
