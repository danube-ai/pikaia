from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pikaia.data.population import PikaiaPopulation


@dataclass(slots=True)
class StrategyContext:
    """
    Holds the context for a strategy calculation.

    Attributes:
        population (Population):
            The population object.
        org_id (int):
            The index of the current organism being evaluated.
        gene_id (int):
            The index of the current gene being evaluated.
        org_fitness (np.ndarray):
            The fitness array for all organisms, shape `(n,)`.
        gene_fitness (np.ndarray):
            The fitness array for all genes, shape `(m,)`.
        org_similarity (np.ndarray):
            A similarity matrix for organisms, shape `(n, n)`.
        gene_similarity (np.ndarray):
            A similarity matrix for genes, shape `(m, m)`.
        initial_org_fitness_range (float):
            The initial range of organism fitness values.

    """

    population: PikaiaPopulation
    org_fitness: np.ndarray
    gene_fitness: np.ndarray
    org_similarity: np.ndarray
    gene_similarity: np.ndarray
    initial_org_fitness_range: float
    org_id: Optional[int] = None
    gene_id: Optional[int] = None


class GeneStrategy(ABC):
    """
    Abstract base class for gene strategies.

    Defines the interface for all gene-level evolutionary strategies. Subclasses
    must implement the `__call__` method, which calculates the fitness delta
    for a specific gene based on the strategy's logic.

    """

    def __init__(self, **kwargs):
        """
        Initializes the strategy with optional parameters.

        Args:
            **kwargs:
                Arbitrary keyword arguments that can be used to configure
                the strategy. These are stored in the `self.options` dictionary.

        """
        self.options = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the strategy."""
        pass

    @abstractmethod
    def __call__(self, ctx: StrategyContext) -> float:
        """
        Computes the delta for a gene strategy.

        Args:
            ctx (StrategyContext):
                Context object containing all required and optional fields.

        Returns:
            float:
                The computed delta value `Delta_G(i,j)` for the specified gene and organism.

        """
        pass


class OrgStrategy(ABC):
    """
    Abstract base class for organism strategies.

    Defines the interface for all organism-level evolutionary strategies.
    Subclasses must implement the `__call__` method, which calculates the
    fitness deltas for all genes based on the organism's interactions.

    """

    def __init__(self, **kwargs):
        """
        Initializes the strategy with optional parameters.

        Args:
            **kwargs:
                Arbitrary keyword arguments that can be used to configure
                the strategy. These are stored in the `self.options` dictionary.

        """
        self.options = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the strategy."""
        pass

    @abstractmethod
    def __call__(self, ctx: StrategyContext) -> np.ndarray:
        """
        Computes deltas for an organism strategy.

        Args:
            ctx (StrategyContext):
                Context object containing all required and optional fields.

        Returns:
            np.ndarray:
                A vector of shape `(m,)` containing the computed
                delta values `Delta_O(i,j)` for the specified organism.

        """
        pass


class MixStrategy(ABC):
    """
    Abstract base class for mixing strategies.

    Defines the interface for strategies that determine how to mix or weigh
    the contributions of different evolutionary strategies (gene or organism).
    Subclasses must implement the `__call__` method.

    """

    def __init__(self, **kwargs):
        """
        Initializes the strategy with optional parameters.

        Args:
            **kwargs:
                Arbitrary keyword arguments that can be used to configure
                the strategy. These are stored in the `self.options` dictionary.

        """
        self.options = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the strategy."""
        pass

    @abstractmethod
    def __call__(
        self, delta: np.ndarray, mix_coeffs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Mixes organism deltas and dynamically updates mixing coefficients.

        The method first calculates the combined delta using the current mixing
        coefficients. It then updates these coefficients for the next iteration.

        Args:
            delta (np.ndarray):
                A 3D array of shape `(n, m, n_strat)` containing the delta matrices
                from each strategy.
            mix_coeffs (np.ndarray):
                A 1D array of shape `(n_strat,)` containing the current mixing
                coefficients for each strategy.

        Returns:
            tuple:
                np.ndarray: The mixed delta matrix of shape `(n, m)`.
                np.ndarray: The updated mixing coefficients for the next iteration.

        """
        pass
