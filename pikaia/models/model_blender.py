"""Provide the high-level Pikaia simulation model and iteration workflow."""

#import multiprocessing

import numpy as np

from pikaia.config.logger import logger
from pikaia.models.pikaia_model import PikaiaModel
# from pikaia.data.population import PikaiaPopulation
# from pikaia.models.genetic_model import GeneticModel
# from pikaia.schemas.strategies import StrategyFormulation
# from pikaia.strategies.base_strategies import (
#     GeneStrategy,
#     OrgStrategy,
#     StrategyContext,
# )


class ModelBlender:
    """tbd

    tbd
    """

    def __init__(self):
        """Initialise the ModelBlender.

        tbd

        Args:
            

        """
        pass

    def blend(self, gamma:np.ndarray, model_list: list, coefficients:np.ndarray
    ) -> np.ndarray:
        """tbd.

        tbd
        """

        step = np.zeros([model_list[0]._population.M])
        for model, model_coeff in zip(model_list, coefficients):
            gene_mix_coeffs = model._gene_mixing_coeffs_hist[-1, :]
            org_mix_coeffs = model._org_mixing_coeffs_hist[-1, :]
            all_pairs = list(
                zip(model._gene_strategies, gene_mix_coeffs)
            ) + list(zip(model._org_strategies, org_mix_coeffs))
            for strat, coeff in all_pairs:
                
                if strat.is_bilinear:
                    step += model_coeff * coeff * gamma * (strat._newDmatrix @ gamma) 
                else:
                    step += model_coeff * coeff * strat._newDmatrix @ gamma 
                #import pdb;pdb.set_trace()

        gamma_new = gamma * (1.0 + step)
        logger.info(
                f"Blended models for new gamma=[{', '.join(f'{x:.6g}' for x in gamma_new.ravel())}]. . "
        )
        return gamma_new