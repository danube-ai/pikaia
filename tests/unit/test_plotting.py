from unittest.mock import patch

import numpy as np
import pytest

from pikaia.data.population import PikaiaPopulation
from pikaia.models.pikaia_model import PikaiaModel
from pikaia.plotting.pikaia_plotter import PikaiaPlotter, PlotType
from pikaia.schemas import GeneStrategyEnum, OrgStrategyEnum
from pikaia.strategies import GeneStrategyFactory, OrgStrategyFactory


class TestPikaiaPlotter:
    """Test cases for PikaiaPlotter class."""

    @pytest.fixture
    def sample_model(self):
        """Create a sample fitted PikaiaModel for testing."""
        # Create sample population
        population = PikaiaPopulation(np.random.rand(5, 3))

        # Create sample strategies
        gene_strategies = [GeneStrategyFactory.get_strategy(GeneStrategyEnum.NONE)]
        org_strategies = [OrgStrategyFactory.get_strategy(OrgStrategyEnum.NONE)]

        # Create and fit model
        model = PikaiaModel(
            population=population,
            gene_strategies=gene_strategies,
            org_strategies=org_strategies,
            max_iter=3,
        )
        model.fit()

        return model

    @pytest.fixture
    def plotter(self, sample_model):
        """Create a PikaiaPlotter instance."""
        return PikaiaPlotter(sample_model)

    def test_plotter_init(self, sample_model):
        """Test PikaiaPlotter initialization."""
        plotter = PikaiaPlotter(sample_model)
        assert plotter.model is sample_model

    def test_plotter_init_unfitted_model(self):
        """Test PikaiaPlotter with unfitted model."""
        population = PikaiaPopulation(np.random.rand(3, 2))
        model = PikaiaModel(population=population)

        # Should work with unfitted model
        plotter = PikaiaPlotter(model)
        assert plotter.model is model

    @pytest.mark.parametrize(
        "plot_type",
        [
            PlotType.GENE_FITNESS_HISTORY,
            PlotType.ORGANISM_FITNESS_HISTORY,
            PlotType.GENE_MIXING_HISTORY,
            PlotType.ORGANISM_MIXING_HISTORY,
            PlotType.GENE_SIMILARITY,
            PlotType.ORGANISM_SIMILARITY,
        ],
    )
    def test_plot_all_types(self, plotter, plot_type):
        """Test plotting all supported plot types."""
        with patch("matplotlib.pyplot.show"):
            fig, ax = plotter.plot(plot_type=plot_type, show=True)
            assert fig is not None
            assert ax is not None

    def test_plot_with_save_path(self, plotter, tmp_path):
        """Test plotting with save path."""
        save_path = tmp_path / "test_plot.png"

        fig, ax = plotter.plot(
            plot_type=PlotType.GENE_FITNESS_HISTORY, save_path=save_path
        )

        assert save_path.exists()
        assert fig is not None
        assert ax is not None

    def test_plot_with_gene_labels(self, plotter):
        """Test plotting with custom gene labels."""
        gene_labels = ["Feature1", "Feature2", "Feature3"]

        with patch("matplotlib.pyplot.show"):
            fig, ax = plotter.plot(
                plot_type=PlotType.GENE_FITNESS_HISTORY,
                gene_labels=gene_labels,
                show=True,
            )
            assert fig is not None
            assert ax is not None

    def test_plot_with_org_labels(self, plotter):
        """Test plotting with custom organism labels."""
        org_labels = ["Org1", "Org2", "Org3", "Org4", "Org5"]

        with patch("matplotlib.pyplot.show"):
            fig, ax = plotter.plot(
                plot_type=PlotType.ORGANISM_FITNESS_HISTORY,
                org_labels=org_labels,
                show=True,
            )
            assert fig is not None
            assert ax is not None

    def test_plot_invalid_type(self, plotter):
        """Test plotting with invalid plot type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid plot type"):
            plotter.plot(plot_type="invalid_type")

    def test_plot_gene_fitness_history_unfitted_model(self):
        """Test gene fitness history plot with unfitted model."""
        population = PikaiaPopulation(np.random.rand(3, 2))
        model = PikaiaModel(population=population)
        plotter = PikaiaPlotter(model)

        # Should handle unfitted model gracefully
        with patch("matplotlib.pyplot.show"):
            fig, ax = plotter.plot(plot_type=PlotType.GENE_FITNESS_HISTORY, show=True)
            assert fig is not None
            assert ax is not None

    def test_plot_organism_fitness_history_unfitted_model(self):
        """Test organism fitness history plot with unfitted model."""
        population = PikaiaPopulation(np.random.rand(3, 2))
        model = PikaiaModel(population=population)
        plotter = PikaiaPlotter(model)

        # Should handle unfitted model gracefully
        with patch("matplotlib.pyplot.show"):
            fig, ax = plotter.plot(
                plot_type=PlotType.ORGANISM_FITNESS_HISTORY, show=True
            )
            assert fig is not None
            assert ax is not None
