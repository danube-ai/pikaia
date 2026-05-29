from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pikaia.data.population import PikaiaPopulation


@dataclass(slots=True)
class StrategyContext:
    """Holds the context for a strategy calculation."""

    #: The population object.
    population: PikaiaPopulation
    #: The fitness array for all organisms, shape ``(n,)``.
    org_fitness: np.ndarray
    #: The fitness array for all genes, shape ``(m,)``.
    gene_fitness: np.ndarray
    #: A similarity matrix for organisms, shape ``(n, n)``.
    org_similarity: np.ndarray
    #: A similarity matrix for genes, shape ``(m, m)``.
    gene_similarity: np.ndarray
    #: The initial range of organism fitness values.
    initial_org_fitness_range: float
    #: The index of the current organism being evaluated.
    org_id: Optional[int] = None
    #: The index of the current gene being evaluated.
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
        pass  # pragma: no cover

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
        pass  # pragma: no cover

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the D-matrix kernel contribution ``(D, d)`` for this strategy.

        ``D`` is an ``(M, M)`` array for the bilinear term
        ``gamma * (D @ gamma)``; ``d`` is an ``(M,)`` array for the linear
        term.  Either may be ``None`` when the strategy has no contribution
        of that type.

        The default returns ``(None, None)`` (zero contribution).  Subclasses
        that support the fast D-matrix path override this method.
        """
        return None, None


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
        pass  # pragma: no cover

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
        pass  # pragma: no cover

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the D-matrix kernel contribution ``(D, d)`` for this strategy.

        The default returns ``(None, None)`` (zero contribution).  Subclasses
        that support the fast D-matrix path override this method.
        """
        return None, None


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
        pass  # pragma: no cover

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
        pass  # pragma: no cover
