#import matplotlib.pyplot as plt
#import matplotlib
import numpy as np
import string

import pikaia
import pikaia.alg
import pikaia.examples
import pikaia.plot

from pikaia.alg import GSStrategy
from pikaia.alg import OSStrategy
        

if __name__ == "__main__":

    # setup numeric printout format
    float_formatter = "{:.4f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    linestyles, markerstylesAltSel, markerstylesDomBal = pikaia.plot.initialize_plotting_variables()

    # set this to the directory where the images should be stored
    relpath = "" # "../../tex/"

    example3x3 = pikaia.examples.assemble_example("3x3-DomBal+AltSal")
    initialgenefitness3x3 = example3x3.exampledata.get_uniform_gene_fitness()
    # change the initial gene fitness for non-symmetric experiments
    # e.g. (should add up to 1)
    # initialgenefitness3x3 = [0.25, 0.5, 0.25]

    # names for output images
    filename3x3gene = relpath + 'Smallexample_3x3_genefitness'
    filename3x3orgs = relpath + 'Smallexample_3x3_orgfitness'

    # real-worlds example 10x5
    example10x5 = pikaia.examples.assemble_example("10x5-DomBal+AltSal")
    initialgenefitness10x5 = example10x5.exampledata.get_uniform_gene_fitness()
    # change the initial gene fitness for non-symmetric experiments
    # # e.g. (should add up to 1)
    # # initialgenefitness10x5 = [0.125, 0.5, 0.125, 0.125, 0.125]

    # names for output images
    filename10x5gene = relpath + 'Realworldexample_10x5_genefitness'
    filename10x5orgs = relpath + 'Realworldexample_10x5_orgfitness'
    
    
    iterations3x3 = 30
    iterations10x5 = 60
    # Using epsilon allows to stop the simulation at ESE
    epsilon = None #0.00005
    
    # define used gene and organism strategies
    strategyDomBal = pikaia.alg.Strategies(GSStrategy.DOMINANT, OSStrategy.BALANCED)
    strategyAltSel = pikaia.alg.Strategies(GSStrategy.ALTRUISTIC, OSStrategy.SELFISH, kinrange=10)
    
    # create models and converge them for simple example 3x3
    dombalSmall = pikaia.alg.Model(example3x3.exampledata, strategyDomBal,
                          example3x3.get_gene_labels(0), example3x3.get_org_labels(0),
                          linestyles, markerstylesDomBal)
    dombalSmall.complete_run(initialgenefitness3x3, iterations3x3, epsilon=epsilon)
    altselSmall = pikaia.alg.Model(example3x3.exampledata, strategyAltSel,
                           example3x3.get_gene_labels(1), example3x3.get_org_labels(1),
                           linestyles, markerstylesAltSel)
    altselSmall.complete_run(initialgenefitness3x3, iterations3x3, epsilon=epsilon)
    pikaia.plot.plot_gene_fitness([dombalSmall, altselSmall], 1, show=True, savename=filename3x3gene)
    pikaia.plot.plot_organism_fitness([dombalSmall, altselSmall], 2, None, show=True, savename=filename3x3orgs)

    # create models and converge them for real-world example 10x5
    dombalLarge = pikaia.alg.Model(example10x5.exampledata, strategyDomBal, 
                          example10x5.get_gene_labels(0), example10x5.get_org_labels(0),
                          linestyles, markerstylesDomBal)
    dombalLarge.complete_run(initialgenefitness10x5, iterations10x5, epsilon=epsilon)

    #import pdb; pdb.set_trace()    
    altselLarge = pikaia.alg.Model(example10x5.exampledata, strategyAltSel,
                          example10x5.get_gene_labels(1), example10x5.get_org_labels(1),
                          linestyles, markerstylesAltSel)
    altselLarge.complete_run(initialgenefitness10x5, iterations10x5, epsilon=epsilon)
    pikaia.plot.plot_gene_fitness([dombalLarge, altselLarge], 1, show=True, savename=filename10x5gene)
     
    maxitershown = 64
    pikaia.plot.plot_organism_fitness([dombalLarge], 2, maxitershown, show=True, savename=filename10x5orgs+"_dombal")
    pikaia.plot.plot_organism_fitness([altselLarge], 2, None, show=True, savename=filename10x5orgs+"_altsel")

           

