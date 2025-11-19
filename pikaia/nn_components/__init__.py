"""Neural network components for Pikaia."""

from .genetic_layer import (
    GeneticLayer,
    GeneticProjection,
    InputProjection,
    OutputProjection,
    StrategyModule,
)
from .mga import MultiHeadGeneticAttention

__all__ = [
    "GeneticLayer",
    "MultiHeadGeneticAttention",
    "InputProjection",
    "GeneticProjection",
    "StrategyModule",
    "OutputProjection",
]
