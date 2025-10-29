import torch
import torch.nn as nn

from pikaia.layers.genetic_layer import GeneticLayer


class ClassicalNet(nn.Module):
    """Classical feed-forward network with residual connections, ReLU, and dropout."""

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        output_size: int,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_features = input_size
        for i in range(num_layers):
            out_features = hidden_size if i < num_layers - 1 else output_size
            self.linears.append(nn.Linear(in_features, out_features))
            if i < num_layers - 1:
                self.relus.append(nn.ReLU())
                self.dropouts.append(nn.Dropout(dropout))
            in_features = out_features

    def forward(self, x):
        for i in range(self.num_layers):
            identity = x
            x = self.linears[i](x)
            if i < self.num_layers - 1:
                x = self.relus[i](x)
                x = self.dropouts[i](x)
                # Residual connection if dimensions match
                if x.shape == identity.shape:
                    x = x + identity
        return x


class GeneticNet(nn.Module):
    """Genetic network with n_heads connected to the same input, concatenated outputs to a final genetic layer."""

    def __init__(
        self,
        input_size: int,
        n_heads: int,
        output_size: int,
        orgs_shape: int = 64,
        genes_shape: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        if n_heads == 1:
            self.head = GeneticLayer(input_size, output_size, genes_shape)
            self.dropout = nn.Dropout(dropout)
        else:
            self.heads = nn.ModuleList(
                [
                    GeneticLayer(input_size, orgs_shape, genes_shape)
                    for _ in range(n_heads)
                ]
            )
            self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_heads)])
            self.final_genetic = GeneticLayer(
                n_heads * orgs_shape, output_size, genes_shape
            )

    def forward(self, x):
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
            output = self.final_genetic(concat)
            return output


def create_classical_network(
    input_size: int,
    num_layers: int,
    output_size: int,
    hidden_size: int = 64,
    dropout: float = 0.1,
):
    """Factory function for classical network."""
    return ClassicalNet(input_size, num_layers, output_size, hidden_size, dropout)


def create_genetic_network(
    input_size: int,
    n_heads: int,
    output_size: int,
    orgs_shape: int = 16,
    genes_shape: int = 8,
    dropout: float = 0.1,
):
    """Factory function for genetic network."""
    return GeneticNet(
        input_size, n_heads, output_size, orgs_shape, genes_shape, dropout
    )
