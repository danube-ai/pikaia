
import numpy as np
import string

import pikaia
import pikaia.alg
import pikaia.examples
import pikaia.plot

from pikaia.alg import GSStrategy
from pikaia.alg import OSStrategy
from pikaia.alg import MixingStrategy
        

if __name__ == "__main__":

    # setup numeric printout format
    float_formatter = "{:.4f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    linestyles, markerstylesAltSel, markerstylesDomBal = pikaia.plot.initialize_plotting_variables()

    # set this to the directory where the images should be stored
    relpath = "" # "../../tex/"

    example3x3 = pikaia.examples.assemble_example("3x3-DomBal+AltSal")
    initialgenefitness3x3 = example3x3.exampledata.get_uniform_gene_fitness()

    # names for output images
    filename3x3gene = relpath + 'SelfConsistentRun_3x3_genefitness'
    filename3x3orgs = relpath + 'SelfConsistentRun_3x3_orgfitness'

    # real-worlds example 10x5
    example10x5 = pikaia.examples.assemble_example("10x5-DomBal+AltSal")
    initialgenefitness10x5 = example10x5.exampledata.get_uniform_gene_fitness()
    # change the initial gene fitness for non-symmetric experiments
    # # e.g. (should add up to 1)
    # # initialgenefitness10x5 = [0.125, 0.5, 0.125, 0.125, 0.125]

    # names for output images
    filename10x5gene = relpath + 'SelfConsistentRun_10x5_genefitness'
    filename10x5mixing = relpath + 'SelfConsistentRun_10x5_mixing'
    
    
    iterations3x3 = 1500
    iterations10x5 = 1500
    # Using epsilon allows to stop the simulation at ESE
    epsilon = None#0.00005
    
    # Comparison: in these calculations we self consistently determine the
    # "right" mixing (sc). The genetic strategies to be mixed are [Dominant, Altruistic]
    # while the organism strategies are [Balanced, Selfish]
    # we start by defining these gene and organism strategies
    strat1 = pikaia.alg.Strategies(GSStrategy.DOMINANT, OSStrategy.BALANCED)
    strat2 = pikaia.alg.Strategies(GSStrategy.ALTRUISTIC, OSStrategy.SELFISH, kinrange=10)
    # For the mixed strategy we use strat1 and strat2 as incredients
    # Note that the initial mixing is uniformly distributed
    strategies = pikaia.alg.Strategies(GSStrategy.MIXED, OSStrategy.MIXED,
                                    kinrange=None,
                                    mixingstrategy=MixingStrategy.SELF_CONSISTENT,
                                    mixinglist=[strat1,strat2],
                                    initialgenemixing=[0.5, 0.5],
                                    initialorgmixing=[0.5, 0.5])
    
    # create models and converge them for simple example 3x3
    mixedSmall = pikaia.alg.Model(example3x3.exampledata, strategies,
                          example3x3.get_gene_labels(2), example3x3.get_org_labels(2),
                          linestyles, markerstylesAltSel)
    mixedSmall.complete_run(initialgenefitness3x3, iterations3x3, epsilon=epsilon)
      

    # We now do the same analysis for the 10x5 flight example
    # Reset Mixing (we reset the initial gene and organism mixing to 0.5)
    strategies = pikaia.alg.Strategies(GSStrategy.MIXED, OSStrategy.MIXED,
                                    kinrange=None,
                                    mixingstrategy=MixingStrategy.SELF_CONSISTENT,
                                    mixinglist=[strat1,strat2],
                                    initialgenemixing=[0.5, 0.5],
                                    initialorgmixing=[0.5, 0.5])
    # create models and converge them for real-world example 10x5
    mixedRealWorld = pikaia.alg.Model(example10x5.exampledata, strategies,
                          example10x5.get_gene_labels(2), example10x5.get_org_labels(2),
                          linestyles, markerstylesDomBal)
    mixedRealWorld.complete_run(initialgenefitness10x5, iterations10x5, epsilon=epsilon)
    
    # Compare the results of gene fitness 
    pikaia.plot.plot_gene_fitness([mixedSmall,  mixedRealWorld], 1, show=True, 
                                  savename=filename10x5gene, postfix=["-(small)", "-(real-world)"])
    # show the convergence of the mixing factors for gene and organism strategies
    pikaia.plot.plot_mixing([mixedSmall, mixedRealWorld], 3, None, 
                            show=True, savename=filename10x5mixing, postfix=["-(small)", "-(real-world)"])
    

           

