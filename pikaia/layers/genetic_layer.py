"""
Genetic Layer Module
"""

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
            Number of latent organisms (output features).
        genes_shape (int):
            Number of latent genes.
        strategy (str):
            The strategy for computing fitness. Currently only 'fixed_point' is
            supported.

    Input:
        x (torch.Tensor):
            Input tensor of shape (..., input_shape), where the last dimension is features.

    Output:
        torch.Tensor:
            Output tensor of shape (..., orgs_shape)
            containing organism fitness values.
    """

    def __init__(
        self,
        input_shape: int,
        orgs_shape: int,
        genes_shape: int,
        strategy: str = "fixed_org_balanced_gene_dominant",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.orgs_shape = orgs_shape
        self.genes_shape = genes_shape
        self.strategy = strategy

        match self.strategy:
            case "fixed_org_balanced_gene_dominant":
                self._run_strategy = self._fixed_org_balanced_gene_dominant_strategy
            case _:
                raise ValueError(
                    f"Unsupported strategy: {self.strategy}. "
                    "Only 'fixed_org_balanced_gene_dominant' is currently supported."
                )

        # Weight matrix
        # (input_shape, orgs_shape * genes_shape)
        self.weight = nn.Parameter(torch.randn(input_shape, orgs_shape * genes_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GeneticLayer.

        Args:
            x (torch.Tensor):
                Input tensor of shape (..., input_shape), where the last dimension is features.

        Returns:
            torch.Tensor:
                Output tensor of shape (..., orgs_shape).
        """
        original_shape = x.shape
        if len(original_shape) < 2:
            raise ValueError(
                "Input tensor must have at least 2 dimensions (batch and features)."
            )

        batch_size = original_shape[0]
        middle_shape = original_shape[1:-1]  # All dims between batch and features
        flattened_middle = (
            int(torch.prod(torch.tensor(middle_shape))) if middle_shape else 1
        )
        input_shape = original_shape[-1]

        # Reshape to 3D: (batch_size, flattened_middle, input_shape)
        x = x.view(batch_size, flattened_middle, input_shape)

        # Proceed with existing logic (now with proper batch_size and input_length)
        input_length = flattened_middle  # Now represents the "sequence" length
        weighted = torch.matmul(x, self.weight)
        population_matrix = weighted.view(
            batch_size, input_length, self.orgs_shape, self.genes_shape
        )
        population_matrix = torch.sigmoid(population_matrix)
        org_fitness = self._run_strategy(population_matrix)

        # Reshape back: (batch_size, ...) + middle_shape + (orgs_shape,)
        output_shape = (batch_size,) + middle_shape + (self.orgs_shape,)
        org_fitness = org_fitness.view(output_shape)

        return org_fitness

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
