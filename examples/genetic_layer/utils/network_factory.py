import math

import torch
import torch.nn as nn
from pikaia.nn_components.genetic_layer import GeneticLayer


def closest_power_of_2(x: float) -> int:
    """
    Find the power of 2 closest to the given number.

    This function calculates the nearest power of 2 to the input value x.
    If x is less than or equal to 0, it returns 1. Otherwise, it compares
    the floor and ceiling powers of 2 and returns the one closer to x.

    Args:
        x (float): The input number to find the closest power of 2 for.

    Returns:
        int: The power of 2 closest to x.

    Examples:
        >>> closest_power_of_2(5)
        4
        >>> closest_power_of_2(7)
        8
    """
    if x <= 0:
        return 1
    log = math.log2(x)
    lower = 2 ** math.floor(log)
    upper = 2 ** math.ceil(log)
    if abs(x - lower) <= abs(x - upper):
        return int(lower)
    else:
        return int(upper)


class ClassicalFeedForwardNet(nn.Module):
    """
    Classical feed-forward neural network with residual connections, SiLU activation,
    layer normalization, and dropout.

    This network consists of multiple linear layers with optional residual connections
    between layers of the same dimension. Each hidden layer includes layer normalization,
    SiLU activation, and dropout for regularization.

    Attributes:
        num_layers (int): Number of layers in the network.
        linears (nn.ModuleList): List of linear layers.
        layer_norms (nn.ModuleList): List of layer normalization modules (for hidden layers).
        silus (nn.ModuleList): List of SiLU activation functions (for hidden layers).
        dropouts (nn.ModuleList): List of dropout modules (for hidden layers).
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        output_size: int,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the ClassicalNet.

        Args:
            input_size (int): Size of the input features.
            num_layers (int): Number of layers in the network (including output layer).
            output_size (int): Size of the output features.
            hidden_size (int, optional): Size of hidden layers. Defaults to 128.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.silus = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_features = input_size
        for i in range(num_layers):
            out_features = hidden_size if i < num_layers - 1 else output_size
            self.linears.append(nn.Linear(in_features, out_features))
            if i < num_layers - 1:
                self.layer_norms.append(nn.LayerNorm(out_features))
                self.silus.append(nn.SiLU())
                self.dropouts.append(nn.Dropout(dropout))
            in_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        for i in range(self.num_layers):
            identity = x
            x = self.linears[i](x)
            if i < self.num_layers - 1:
                x = self.layer_norms[i](x)
                x = self.silus[i](x)
                x = self.dropouts[i](x)
                # Residual connection if dimensions match
                if x.shape == identity.shape:
                    x = x + identity
        return x


class ClassicalHeadsNet(nn.Module):
    """
    Classical heads-based neural network with multiple heads connected to the same input,
    concatenating outputs to a final linear layer.

    Similar to GeneticNet but uses standard linear layers instead of genetic layers.

    Attributes:
        n_heads (int): Number of heads.
        head (nn.Linear): Single linear head (if n_heads == 1).
        dropout (nn.Dropout): Dropout for single head.
        heads (nn.ModuleList): List of linear heads (if n_heads > 1).
        dropouts (nn.ModuleList): List of dropouts for each head.
        final_linear (nn.Linear): Final linear layer for concatenation.
    """

    def __init__(
        self,
        input_size: int,
        n_heads: int,
        output_size: int,
        hidden_size: int = 32,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the ClassicalHeadsNet.

        Args:
            input_size (int): Size of the input features.
            n_heads (int): Number of heads.
            output_size (int): Size of the output features.
            hidden_size (int, optional): Hidden size for heads. Defaults to 32.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.n_heads = n_heads
        if n_heads == 1:
            self.head = nn.Linear(input_size, output_size)
            self.dropout = nn.Dropout(dropout)
        else:
            self.heads = nn.ModuleList(
                [nn.Linear(input_size, hidden_size) for _ in range(n_heads)]
            )
            self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_heads)])
            self.final_linear = nn.Linear(n_heads * hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        if self.n_heads == 1:
            output = self.head(x)
            output = self.dropout(output)
            return output
        else:
            head_outputs = []
            for i in range(self.n_heads):
                out = self.heads[i](x)
                out = self.dropouts[i](out)
                head_outputs.append(out)
            concat = torch.cat(head_outputs, dim=-1)
            output = self.final_linear(concat)
            return output


class GeneticHeadsNet(nn.Module):
    """
    Genetic neural network with multiple heads connected to the same input,
    concatenating outputs to a final linear layer.

    This network uses GeneticLayer instances as heads. For single head, it directly
    outputs through the genetic layer. For multiple heads, it concatenates the outputs
    and passes through a final linear layer.

    Attributes:
        n_heads (int): Number of genetic heads.
        head (GeneticLayer): Single genetic head (if n_heads == 1).
        dropout (nn.Dropout): Dropout for single head.
        heads (nn.ModuleList): List of genetic heads (if n_heads > 1).
        dropouts (nn.ModuleList): List of dropouts for each head.
        final_linear (nn.Linear): Final linear layer for concatenation.
    """

    def __init__(
        self,
        input_size: int,
        n_heads: int,
        output_size: int,
        orgs_shape: int = 6,
        genes_shape: int = 3,
        dropout: float = 0.15,
        hidden_size: int = 32,
    ) -> None:
        """
        Initialize the GeneticNet.

        Args:
            input_size (int): Size of the input features.
            n_heads (int): Number of genetic heads.
            output_size (int): Size of the output features.
            orgs_shape (int, optional): Shape parameter for organisms. Defaults to 6.
            genes_shape (int, optional): Shape parameter for genes. Defaults to 3.
            dropout (float, optional): Dropout probability. Defaults to 0.15.
            hidden_size (int, optional): Hidden size for heads. Defaults to 32.
        """
        super().__init__()
        self.n_heads = n_heads
        if n_heads == 1:
            self.head = GeneticLayer(
                input_size,
                hidden_size,
                orgs_shape,
                genes_shape,
                output_shape=output_size,
            )
            self.dropout = nn.Dropout(dropout)
        else:
            self.heads = nn.ModuleList(
                [
                    GeneticLayer(
                        input_size,
                        hidden_size,
                        orgs_shape,
                        genes_shape,
                        output_shape=hidden_size,
                    )
                    for _ in range(n_heads)
                ]
            )
            self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_heads)])
            self.final_linear = nn.Linear(n_heads * hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        if self.n_heads == 1:
            output = self.head(x)
            output = self.dropout(output)
            return output
        else:
            head_outputs = []
            for i in range(self.n_heads):
                out = self.heads[i](x)
                out = self.dropouts[i](out)
                head_outputs.append(out)
            concat = torch.cat(head_outputs, dim=-1)
            output = self.final_linear(concat)
            return output


class GeneticFeedForwardNet(nn.Module):
    """
    Genetic feed-forward neural network with sequential genetic layers.

    This network stacks multiple GeneticLayer instances sequentially with dropout for regularization.
    Depth is limited to prevent gradient issues but increased from previous version.

    Attributes:
        num_layers (int): Number of genetic layers (max 5).
        genetic_layers (nn.ModuleList): List of GeneticLayer instances.
        dropouts (nn.ModuleList): List of dropout modules between layers.
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        output_size: int,
        orgs_shape: int = 4,
        genes_shape: int = 2,
        dropout: float = 0.2,
        hidden_size: int = 32,
    ) -> None:
        """
        Initialize the GeneticFeedForwardNet.

        Args:
            input_size (int): Size of the input features.
            num_layers (int): Number of genetic layers (automatically capped at 5).
            output_size (int): Size of the output features.
            orgs_shape (int, optional): Shape parameter for organisms. Defaults to 4.
            genes_shape (int, optional): Shape parameter for genes. Defaults to 2.
            dropout (float, optional): Dropout probability. Defaults to 0.2.
            hidden_size (int, optional): Hidden size for layers. Defaults to 32.
        """
        super().__init__()
        # Limit depth to prevent gradient issues but allow deeper networks
        self.num_layers = min(num_layers, 5)
        self.genetic_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_features = input_size
        for i in range(self.num_layers):
            out_features = hidden_size if i < self.num_layers - 1 else output_size
            self.genetic_layers.append(
                GeneticLayer(
                    in_features,
                    hidden_size,
                    orgs_shape,
                    genes_shape,
                    output_shape=out_features,
                    dropout_rate=dropout,
                )
            )
            if i < self.num_layers - 1:
                self.dropouts.append(nn.Dropout(dropout))
            in_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        for i in range(self.num_layers):
            x = self.genetic_layers[i](x)
            if i < self.num_layers - 1:
                x = self.dropouts[i](x)
        return x


def create_classical_network(
    input_size: int,
    num_layers: int,
    output_size: int,
    hidden_size: int = 64,
    dropout: float = 0.1,
) -> ClassicalFeedForwardNet:
    """
    Factory function to create a classical neural network.

    Args:
        input_size (int): Size of the input features.
        num_layers (int): Number of layers in the network.
        output_size (int): Size of the output features.
        hidden_size (int, optional): Size of hidden layers. Defaults to 64.
        dropout (float, optional): Dropout probability. Defaults to 0.1.

    Returns:
        ClassicalNet: The created classical network instance.
    """
    return ClassicalFeedForwardNet(
        input_size, num_layers, output_size, hidden_size, dropout
    )


def create_genetic_network(
    input_size: int,
    n_heads: int,
    output_size: int,
    orgs_shape: int = 4,
    genes_shape: int = 2,
    dropout: float = 0.2,
    hidden_size: int = 32,
) -> GeneticHeadsNet:
    """
    Factory function to create a genetic neural network.

    Args:
        input_size (int): Size of the input features.
        n_heads (int): Number of genetic heads.
        output_size (int): Size of the output features.
        orgs_shape (int, optional): Shape parameter for organisms. Defaults to 4.
        genes_shape (int, optional): Shape parameter for genes. Defaults to 2.
        dropout (float, optional): Dropout probability. Defaults to 0.2.
        hidden_size (int, optional): Hidden size for heads. Defaults to 32.

    Returns:
        GeneticNet: The created genetic network instance.
    """
    return GeneticHeadsNet(
        input_size,
        n_heads,
        output_size,
        orgs_shape,
        genes_shape,
        dropout,
        hidden_size,
    )


def create_feedforward_classical_network(
    input_size: int,
    depth: int,
    output_size: int,
    hidden_size: int = 64,
    dropout: float = 0.1,
) -> ClassicalFeedForwardNet:
    """
    Factory function to create a feed-forward classical neural network.

    Args:
        input_size (int): Size of the input features.
        depth (int): Number of layers (depth) in the network.
        output_size (int): Size of the output features.
        hidden_size (int, optional): Size of hidden layers. Defaults to 64.
        dropout (float, optional): Dropout probability. Defaults to 0.1.

    Returns:
        ClassicalNet: The created feed-forward classical network instance.
    """
    return ClassicalFeedForwardNet(input_size, depth, output_size, hidden_size, dropout)


def create_heads_classical_network(
    input_size: int,
    n_heads: int,
    output_size: int,
    hidden_size: int = 32,
    dropout: float = 0.1,
) -> ClassicalHeadsNet:
    """
    Factory function to create a heads-based classical neural network.

    Args:
        input_size (int): Size of the input features.
        n_heads (int): Number of heads.
        output_size (int): Size of the output features.
        hidden_size (int, optional): Hidden size for heads. Defaults to 32.
        dropout (float, optional): Dropout probability. Defaults to 0.1.

    Returns:
        ClassicalHeadsNet: The created heads-based classical network instance.
    """
    return ClassicalHeadsNet(input_size, n_heads, output_size, hidden_size, dropout)


def create_feedforward_genetic_network(
    input_size: int,
    depth: int,
    output_size: int,
    orgs_shape: int = 4,
    genes_shape: int = 2,
    dropout: float = 0.2,
    hidden_size: int = 32,
) -> GeneticFeedForwardNet:
    """
    Factory function to create a feed-forward genetic neural network.

    Args:
        input_size (int): Size of the input features.
        depth (int): Number of genetic layers (depth).
        output_size (int): Size of the output features.
        orgs_shape (int, optional): Shape parameter for organisms. Defaults to 4.
        genes_shape (int, optional): Shape parameter for genes. Defaults to 2.
        dropout (float, optional): Dropout probability. Defaults to 0.2.
        hidden_size (int, optional): Hidden size for layers. Defaults to 32.

    Returns:
        GeneticFeedForwardNet: The created feed-forward genetic network instance.
    """
    return GeneticFeedForwardNet(
        input_size, depth, output_size, orgs_shape, genes_shape, dropout, hidden_size
    )


def create_heads_genetic_network(
    input_size: int,
    n_heads: int,
    output_size: int,
    orgs_shape: int = 6,
    genes_shape: int = 3,
    dropout: float = 0.15,
    hidden_size: int = 32,
) -> GeneticHeadsNet:
    """
    Factory function to create a heads-based genetic neural network.

    Args:
        input_size (int): Size of the input features.
        n_heads (int): Number of genetic heads.
        output_size (int): Size of the output features.
        orgs_shape (int, optional): Shape parameter for organisms. Defaults to 6.
        genes_shape (int, optional): Shape parameter for genes. Defaults to 3.
        dropout (float, optional): Dropout probability. Defaults to 0.15.
        hidden_size (int, optional): Hidden size for heads. Defaults to 32.

    Returns:
        GeneticNet: The created heads-based genetic network instance.
    """
    return GeneticHeadsNet(
        input_size,
        n_heads,
        output_size,
        orgs_shape,
        genes_shape,
        dropout,
        hidden_size,
    )
