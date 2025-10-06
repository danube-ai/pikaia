from enum import StrEnum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pikaia.models.pikaia_model import PikaiaModel


class PlotType(StrEnum):
    """
    Enum for the different types of plots.
    """

    GENE_FITNESS_HISTORY = "gene_fitness_history"
    ORGANISM_FITNESS_HISTORY = "organism_fitness_history"
    GENE_MIXING_HISTORY = "gene_mixing_history"
    ORGANISM_MIXING_HISTORY = "organism_mixing_history"
    GENE_SIMILARITY = "gene_similarity"
    ORGANISM_SIMILARITY = "organism_similarity"


class PikaiaPlotter:
    """
    A class for plotting results from a PikaiaModel.

    This class provides a set of methods to visualize the outputs of an evolutionary
    simulation, including fitness histories, mixing coefficients, and similarity matrices.

    """

    def __init__(self, model: PikaiaModel):
        """
        Initializes the PikaiaPlotter with a PikaiaModel instance.

        Args:
            model (PikaiaModel):
                The fitted PikaiaModel to be plotted.

        """
        self.model = model
        plt.style.use("seaborn-v0_8-whitegrid")

    def plot(
        self,
        plot_type: PlotType,
        show: bool = False,
        save_path: Path | None = None,
        gene_labels: list[str] | None = None,
        org_labels: list[str] | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Plots the specified data from the model.

        Args:
            plot_type (PlotType):
                The type of plot to generate.
            show (bool):
                If True, the plot is displayed. Defaults to False.
            save_path (Path | None):
                If provided, the plot is saved to this path. Defaults to None.
            gene_labels (list[str] | None):
                Custom labels for genes.
            org_labels (list[str] | None):
                Custom labels for organisms.

        Returns:
            tuple:
                A tuple containing the matplotlib Figure and Axes objects.

        """
        if gene_labels is None:
            gene_labels = [f"Gene {i}" for i in range(self.model.population.M)]
        if org_labels is None:
            org_labels = [f"Organism {i}" for i in range(self.model.population.N)]

        match plot_type:
            case PlotType.GENE_FITNESS_HISTORY:
                fig, ax = self._plot_history(
                    data=self.model.gene_fitness_history,
                    title="Gene Fitness History",
                    ylabel="Fitness",
                    labels=gene_labels,
                    show=show,
                    save_path=save_path,
                )
            case PlotType.ORGANISM_FITNESS_HISTORY:
                fig, ax = self._plot_history(
                    data=self.model.organism_fitness_history,
                    title="Organism Fitness History",
                    ylabel="Fitness",
                    labels=org_labels,
                    show=show,
                    save_path=save_path,
                )
            case PlotType.GENE_MIXING_HISTORY:
                fig, ax = self._plot_history(
                    data=self.model.gene_mixing_history,
                    title="Gene Mixing Coefficients History",
                    ylabel="Mixing Coefficient",
                    labels=[strategy.name for strategy in self.model.gene_strategies],
                    show=show,
                    save_path=save_path,
                )
            case PlotType.ORGANISM_MIXING_HISTORY:
                fig, ax = self._plot_history(
                    data=self.model.organism_mixing_history,
                    title="Organism Mixing Coefficients History",
                    ylabel="Mixing Coefficient",
                    labels=[strategy.name for strategy in self.model.org_strategies],
                    show=show,
                    save_path=save_path,
                )
            case PlotType.GENE_SIMILARITY:
                fig, ax = self._plot_heatmap(
                    data=self.model.gene_similarity,
                    title="Gene Similarity Matrix",
                    labels=gene_labels,
                    show=show,
                    save_path=save_path,
                )
            case PlotType.ORGANISM_SIMILARITY:
                fig, ax = self._plot_heatmap(
                    data=self.model.org_similarity,
                    title="Organism Similarity Matrix",
                    labels=org_labels,
                    show=show,
                    save_path=save_path,
                )
            case _:
                raise ValueError(f"Invalid plot type: {plot_type}")
        return fig, ax

    def _plot_history(
        self,
        data: np.ndarray,
        title: str,
        ylabel: str,
        labels: list[str] | None = None,
        show: bool = False,
        save_path: Path | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Helper function to plot 2D history data.

        Args:
            data (np.ndarray):
                The 2D data array to plot (iterations x variables).
            title (str):
                The title of the plot.
            ylabel (str):
                The label for the y-axis.
            labels (list[str] | None, optional):
                Labels for each line. Defaults to None.
            show (bool):
                Whether to display the plot. Defaults to False.
            save_path (Path | None, optional):
                Filename to save the plot. Defaults to None.

        Returns:
            tuple:
                A tuple containing the matplotlib Figure and Axes objects.

        """
        fig, ax = plt.subplots(figsize=(10, 6))
        num_iterations, num_vars = data.shape
        iterations = range(num_iterations)

        for i in range(num_vars):
            label = labels[i] if labels else f"Variable {i + 1}"
            ax.plot(iterations, data[:, i], marker="o", linestyle="-", label=label)

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend()
        ax.grid(True)

        if save_path:
            plt.savefig(save_path.with_suffix(".png"), bbox_inches="tight", dpi=300)

        if show:
            plt.show()
        elif save_path is not None:
            plt.close(fig)
        return fig, ax

    def _plot_heatmap(
        self,
        data: np.ndarray,
        title: str,
        labels: list[str] | None = None,
        show: bool = False,
        save_path: Path | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Helper function to plot a similarity matrix as a heatmap.

        Args:
            data (np.ndarray):
                The similarity matrix to plot.
            title (str):
                The title of the plot.
            labels (list[str] | None, optional):
                Labels for the ticks. Defaults to None.
            show (bool):
                Whether to display the plot. Defaults to False.
            save_path (Path | None, optional):
                Filename to save the plot. Defaults to None.

        Returns:
            tuple:
                A tuple containing the matplotlib Figure and Axes objects.

        """
        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.matshow(data, cmap="viridis")
        fig.colorbar(cax)

        ax.set_title(title, fontsize=16, pad=20)

        if labels:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=90)
            ax.set_yticklabels(labels)

        if save_path:
            plt.savefig(save_path.with_suffix(".png"), bbox_inches="tight", dpi=300)

        if show:
            plt.show()
        return fig, ax
