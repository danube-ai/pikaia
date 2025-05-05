"""Module containing a main function that instantiates a Genetic AI
model and runs a small evolutionary simulation with pikaia and a
minimal dataset.
"""
import numpy as np

import pikaia

from pikaia.alg import GVRule
from pikaia.alg import GSStrategy
from pikaia.alg import OSStrategy
from pikaia.alg import MixingStrategy

if __name__ == "__main__":
    # setup numeric printout format
    float_formatter = "{:.4f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    # setup example data
    # rows are the organisms (length n), columns are the genes (length m)
    rawdata = np.zeros([3,3])
    rawdata[:,:] = [[ 300, 10, 2],
                    [ 600,  5, 2],
                    [1500,  4, 1]]

    # define gene fitness rules for every gene separately
    # length needs to be m
    gvfitnessrules = [GVRule.PERCENTAGE_INVERTED,
                      GVRule.PERCENTAGE_INVERTED,
                      GVRule.PERCENTAGE_INVERTED]

    # instantiate population object; this will derive population data
    # from the raw data by applying gene variant fitness functions
    population = pikaia.alg.Population(rawdata, gvfitnessrules)

    # define the strategies used when running the evolutionary simulation;
    # you can pre-define the strategies depending on your use case or let
    # pikaia select the best strategy mix depending on your data

    # Pre-defined strategies
    strat1 = pikaia.alg.Strategies(GSStrategy.DOMINANT, OSStrategy.BALANCED)
    strat2 = pikaia.alg.Strategies(GSStrategy.ALTRUISTIC, OSStrategy.SELFISH, kinrange=10)

    # Mix the strategies equally
    strategies = pikaia.alg.Strategies(GSStrategy.MIXED, OSStrategy.MIXED,
                                    kinrange=None,
                                    mixingstrategy=MixingStrategy.FIXED,
                                    mixinglist=[strat1,strat2],
                                    initialgenemixing=[0.5, 0.5],
                                    initialorgmixing=[0.5, 0.5])

    iterations = 1 # Number of iterations

    # create the genetic model
    model = pikaia.alg.Model(population, strategies)

    # define initial gene fitness values; these are usually just
    # (1/m) for every gene.
    initialgenefitness = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    model.complete_run(initialgenefitness, iterations)
