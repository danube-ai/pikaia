import numpy as np
from enum import Enum

class GVRule(Enum):
    PERCENTAGE = 1
    PERCENTAGE_INVERTED = 2
    PERCENTAGE_CAPPED = 3
    PERCENTAGE_INVERTED_CAPPED = 4
class GSStrategy(Enum):
    NONE = 0
    DOMINANT = 1
    ALTRUISTIC = 2
    MIXED = 3
    SELFISH = 4
    KIN_ALTRUISTIC = 5

class OSStrategy(Enum):
    NONE = 0
    BALANCED = 1
    SELFISH = 2
    MIXED = 3    
    ALTRUISTIC = 4
    KIN_SELFISH = 5

class MixingStrategy(Enum):
    NONE = 0
    FIXED = 1
    SELF_CONSISTENT = 2

class Model:
    """Central organizing class for the model in Genetic AI.

    The Model class contains the population, the used evolutionary
    strategies and auxiliary labeling and plotting information on 
    genes and organisms.

    Args:
        population (pikaia.alg.Population): The input data in the form
            of normalized floating gene variant values. The population
            contains n organisms with m genes.
        strategies(pikaia.alg.Strategies): The used gene and organism
            strategies.
        genelabels(list of strings, optional): The gene labels used 
            for plotting. Length should be m.
        orglabels(list of strings, optional): The organism labels used 
            for plotting.
        linestyles(list of strings, optional): The linestyle strings for
            plotting
        markerstyles(list of strings, optional): The markerstyle strings for
            plotting
    """

    def __init__(self, population, strategies, 
                 genelabels=None, orgslabels=None,
                 linestyles=None, markerstyles=None):
        self._population = population
        self._strategies = strategies
        self._genelabels = genelabels
        self._orgslabels = orgslabels
        self._linestyles = linestyles
        self._markerstyles = markerstyles

    @property
    def population(self):
        """Returns the population used in the model."""
        return self._population
    
    @property
    def strategies(self):
        """Returns the strategies used in the model."""
        return self._strategies

    def complete_run(self, initialgenefitness, maxiteration=1, epsilon=None, silent=False):
        """Runs an evolutionary simulation.

        Runs a evolutionary simulation until either maxiter 
        is reached or the difference of gene fitness values
        from two consecutive iterations are below epsilon.

        Args:
            initialgenefitness (vector of shape `(1, m)`):
                The initial fitness used in this simulation.
            maxiteration (int, optional): if given, defines
                the maximum number of iterations of evolutionary
                simulation.
            epsilon (float, optional): If given, defines 
                the breaking condition for the simulation.
            silent (boolean, optional): Setting to true
                deactivates most printouts.
        
        Returns:
            the similarity matrix `(n, n)`,
        """
        self._maxiter = maxiteration
        self._epsilon = epsilon
        genefitness = initialgenefitness
        print("initial gene fitness = ", genefitness)
        initialorganismfitness = compute_all_organism_fitness(self.population.data, genefitness)
        if not silent:
            print("initial organism fitness = ", initialorganismfitness)
        initialorgfitnessrange = max(initialorganismfitness) - min(initialorganismfitness)
        if max(initialorganismfitness) == 0:
            raise ValueError("All organism fitness values are 0.")
        elif initialorgfitnessrange == 0:#if all fitness values are equal take max instead
            initialorgfitnessrange = max(initialorganismfitness)
        
        #iter = self._maxiter
        n = self.population.data.shape[0]
        m = self.population.data.shape[1]
        nstrat = len(self._strategies.mixinglist)
        # helper structures storing iterations
        self._gps = np.zeros([self._maxiter+1, m])
        self._ofs = np.zeros([self._maxiter+1, n])
        self._genemixing = np.zeros([self._maxiter+1, nstrat])
        self._orgmixing = np.zeros([self._maxiter+1, nstrat])
        self._gps[0,:] = genefitness
        self._ofs[0,:] = initialorganismfitness
        self._genemixing[0,:] = self._strategies.initialgenemixing
        self._orgmixing[0,:] = self._strategies.initialorgmixing

        # compute similarities necessary for some strategies
        genesimilarity = compute_gene_similarity(self.population.data)
        orgsimilarity = compute_organism_similarity(self.population.data)
        
        for k in range(0, self._maxiter):
            if not silent:
                print("+++ Iteration ", k," +++")
            genefitness, fitness = iteration(self.population.data, genefitness,
                                            genesimilarity, orgsimilarity,
                                            initialorgfitnessrange,
                                            self._strategies, silent=silent)
            self._gps[k+1,:] = genefitness
            self._ofs[k+1,:] = fitness
            self._genemixing[k+1,:] = self._strategies.currentgenemixing
            self._orgmixing[k+1,:] = self._strategies.currentorgmixing
            if not silent:
                print("gene fitness = ", genefitness)
                print("organism fitness = ", fitness)
            delta = np.linalg.norm(self._gps[k+1,:]-self._gps[k,:])
            self._iterESE = k + 1
            if self._epsilon is not None and delta < self._epsilon:
                print("reached ESE after ", self._iterESE, "iterations. Final delta = ", delta)
                break
        return genefitness, fitness

    
class Strategies:
    """Holds gene, organism and mixing strategies.

    Args:
        gsstrategy (enum): used gene strategy.
        osstrategy (enum): used organism strategy.
        kinrange (int, optional): number of relatives used for
            certain strategies.
        mixingstrategy (enum, optional): used mixing strategy.
        mixinglist (list, optional): list of Strategy class instances
            to be mixed
        initialgenemixing (list of floats, optional): initial mixing coefficients
            for gene mixing. Should be same length as mixinglist.
        initialorgmixing (list of floats, optional): initial mixing coefficients
            for organism mixing. Should be same length as mixinglist.
    """
    def __init__(self, gsstrategy, osstrategy, kinrange=None,
                 mixingstrategy=MixingStrategy.NONE, 
                 mixinglist=None,
                 initialgenemixing=None, initialorgmixing=None):
        self._gsstrategy = gsstrategy
        self._osstrategy = osstrategy
        self._mixingstrategy = mixingstrategy
        self._kinrange = kinrange
        if mixinglist is None:
            self._mixinglist = [self]
            self._initialgenemixing = [1]
            self._currentgenemixing = [1]
            self._initialorgmixing = [1]
            self._currentorgmixing = [1]
        else:
            self._mixinglist = mixinglist
            self._initialgenemixing = initialgenemixing
            self._currentgenemixing = self._initialgenemixing
            self._initialorgmixing = initialorgmixing
            self._currentorgmixing = self._initialorgmixing

    @property
    def gsstrategy(self):
        """Returns gene strategy enum"""
        return self._gsstrategy
    
    @property
    def osstrategy(self):
        """Returns organism strategy enum"""
        return self._osstrategy
    
    @property
    def mixingstrategy(self):
        """Returns mixing strategy enum"""
        return self._mixingstrategy
    
    @property
    def mixinglist(self):
        """Returns list of mixed strategies"""
        return self._mixinglist
    
    @property
    def initialgenemixing(self):
        """Returns initial coefficients for gene mixing"""
        return self._initialgenemixing

    @property
    def currentgenemixing(self):
        """Returns current gene mixing coefficients"""
        return self._currentgenemixing
    
    
    @property
    def initialorgmixing(self):
        """Returns initial organism mixing coefficients"""
        return self._initialorgmixing

    @property
    def currentorgmixing(self):
        """Returns current organism mixing coefficients"""
        return self._currentorgmixing
    
    @property
    def kinrange(self):
        """Returns number of used relatives"""
        if self._kinrange is None:
            raise ValueError("Property 'kinrange' not defined")
        else:
            return self._kinrange

def apply_deltas(oldvalues, deltas):
    """Applies delta in form of the central replicator equations.

    The are two steps in applying the replicator equations: (i)
    First the deltas are applied as newval = val*(1+delta). (ii)
    Second the deltas are normalized such that the sum equals 1.
    For this to work consistently the deltas should not be below
    -1.

    Args:
        oldvalues (vector of float): Old values values to be
            deltered.
        delta (vector of floats): Applied delta values. Should
            be the same length as oldvalues.
        
    """
    nv = len(oldvalues)
    newvalues = np.zeros(nv)
    for t in range(0, nv):
        newvalues[t] = oldvalues[t]*(1+deltas[t])
    sumvalues = sum(newvalues)
    for t in range(0, nv):
        newvalues[t] /= sumvalues
    return newvalues
class Population:
    """Contains the normalized input data for a Genetic AI model.

    The raw input data is converted to values in [0,1] following
    the given gene fitness rules.

    Args:
        rawdata (matrix of shape `(n, m)`): A matrix containing
            structured data.
        gvfitnessrule (list): Defines how the raw data is converted
            to a population. The conversion is done implicitly on
            creation of the instance. Should have length m.
    """
    def __init__(self, rawdata, gvfitnessrule, silent=False):
        self._rawdata = rawdata
        self._n = self._rawdata.shape[0] #number of organisms
        self._m = self._rawdata.shape[1] #number of genes
        self._gvfitnessrules = gvfitnessrule
        self._populationdata = np.zeros([self._n,self._m]) 
        for j in range(0, self._m):
            # the populations derive from the raw data by applying gene variant 
            # fitness functions
            self._populationdata[:,j] = compute_gene_variant_fitness(self._rawdata[:,j], 
                                                                 self._gvfitnessrules[j])
        if not silent:
            print("Population = ", self._populationdata)

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m

    @property
    def rawdata(self):
        return self._rawdata
    
    @property
    def data(self):
        return self._populationdata
    
    def get_uniform_gene_fitness(self):
        genefitness = np.zeros([self._m])
        for j in range(0, self._m):
            genefitness[j] = 1.0/self._m
        return genefitness
       

def compute_gene_variant_fitness(genedata, gvfrule):
    """Computes the gene variant fitness from raw data.

    Args:
        genedata (vector of shape `(n, 1)`):
            The input data.
        gvrule (enum):
            The gene variant fitness function applied.

    Returns:
        A vector of shape `(n, 1)`, the gene variant fitness values.
    """
    if gvfrule == GVRule.PERCENTAGE_INVERTED or \
       gvfrule == GVRule.PERCENTAGE_INVERTED_CAPPED:
        invert = True
    else:
        invert = False
    
    if gvfrule == GVRule.PERCENTAGE_CAPPED or \
       gvfrule == GVRule.PERCENTAGE_INVERTED_CAPPED:
        cap = True
    else:
        cap = False

    n = len(genedata)
    maxvariant = max(genedata)
    minvariant = min(genedata)
    gvfitness = np.zeros(n)
    for i in range(0, n):
        if invert:
            if cap:
                gvfitness[i] = (maxvariant - genedata[i])/(maxvariant-minvariant) if (maxvariant-minvariant) != 0 else 0
            else:
                gvfitness[i] = (maxvariant - genedata[i])/maxvariant if maxvariant > 0 else 0
        else:
            if cap:
                gvfitness[i] = genedata[i]/(maxvariant-minvariant) if maxvariant > 0 else 0
            else:
                gvfitness[i] = genedata[i]/maxvariant if maxvariant > 0 else 0
    return gvfitness
        

def compute_organism_fitness(organism, genefitness):
    """Computes the (linear) organism fitness.
    
    Dot product of organism and genefitness.

    Args:
        organism (vector of shape `(1, m)`):
            The gene variant fitness values of a single organism.
        genefitness (vector of shape `(1, m)`):
            The list of gene fitness values.
    
    Returns:
        A float, the linear organism fitness.
    """
    ofitness = 0.0
    for j in range(0, len(organism)):
        ofitness += organism[j]*genefitness[j]
    return ofitness

def compute_all_organism_fitness(population, genefitness):
    """Computes the (linear) organism fitness values for a population.

    Args:
        population (matrix of shape `(n, m)`):
            The population data in matrix form, rows are organisms,
            colums genes.
        genefitness (vector of shape `(1, m)`):
            The list of gene fitness values.
    
    Returns:
        A vector of shape `(n, 1)`, the list of fitness values
        of the organisms in the population.
    """
    n = population.shape[0]
    populationfitness = np.zeros(n)
    for i in range(0, n):
        populationfitness[i] = compute_organism_fitness(population[i,:], genefitness)
    return populationfitness


def compute_gene_deltas(geneindex, genevariantfitness, gene, organism, genefitness,  
                        genesimilarity, strategies):
    """Computes delta values stemming from the gene strategy.

    Args:
        geneindex (integer):
            The index j for which gene index the delta is to be computed.
        genevariantfitness (float):
            the gene variant fitness at i,j.
        gene (float vector of shape `(1, n)`):
            the current gene vector.
        organism (float vector of shape `(m, 1)`):
            the current organism vector.
        genefitness (float vector of shape `(1, m)`):
            The list of gene fitness values.
        genesimilarity (float vector of shape `(1, m)`):
            The kinship of the current gene to the others.
        strategies (pikaia.alg.Strategies):
            The evolutionary strategy to be applied.
         
    Returns:
        A float, the Delta(i,j) value for a particular gene 
        and organism, respectively.
    """    
    n = len(gene)
    m = len(organism)
    if strategies.gsstrategy == GSStrategy.DOMINANT:
        # factors 4 (gene fitness x 2)
        deltaG = 4/float(n)*genefitness[geneindex]*genefitness[geneindex]*(genevariantfitness-1/2)
    elif strategies.gsstrategy == GSStrategy.SELFISH: # Experimental!
        deltaG = 0
        for j in range(0, m):
            if j == geneindex:
                continue
            deltaG += -16/float(n)*genesimilarity[j]*genefitness[geneindex]*(genevariantfitness-1/2)*\
                                 genefitness[j]*(organism[j] - genevariantfitness)
        deltaG  = deltaG/float(m)
    elif strategies.gsstrategy == GSStrategy.KIN_ALTRUISTIC:# Experimental!
        deltaG = 0
        for j in range(0, m):
            if j == geneindex:
                continue
            deltaG += 16/float(n)*(0.5-genesimilarity[j])*genefitness[geneindex]*(genevariantfitness-1/2)*\
                                genefitness[j]*(organism[j] - genevariantfitness)
        deltaG  = deltaG/float(m)
    elif strategies.gsstrategy == GSStrategy.ALTRUISTIC:
        deltaG = 0
        for j in range(0, m):
            if j == geneindex:
                continue
            # factors: 2 (genesimilarity) x 4 (genefitness x 2) x 2 (comparison)
            deltaG += 16/float(n)*genesimilarity[j]*genefitness[geneindex]*(genevariantfitness-1/2)*\
                                 genefitness[j]*(organism[j] - genevariantfitness)
            # print(organism, geneindex,j, deltaG/float(m), 8/float(n)/float(m)*genefitness[geneindex]*(genevariantfitness-1/2),
            #                    8/float(n)/float(m)*genesimilarity[j]*\
            #                      genefitness[j]*(organism[j] - genevariantfitness))
            
        deltaG  = deltaG/float(m)
    elif strategies.gsstrategy == GSStrategy.NONE:
        deltaG = 0
    else:
        raise NameError("Unknown gene strategy:" + str(strategies.gsstrategy))
    
    return deltaG

def mix_gene_deltas(deltaGtmp, strategies, silent=False):
    """Mixes delta-contributions from multiple gene strategies.

    Note that if only one strategy is used, this function simply
    returns the input matrix as is.

    Args:
        deltaGtmp (matrix of shape `(n, m, nstrat)`):
            The delta-matrices for the to-be-mixed strategies.        
        strategies (pikaia.alg.Strategies):
            Contains the mixing strategy, either {fixed, self-consistent}.
    
    Returns:
        A matrix of shape `(n, m)` containing the mixed gene delta matrix.
    """ 

    n = deltaGtmp.shape[0]
    m = deltaGtmp.shape[1]
    
    deltaG = np.zeros([n,m])
    
    for s, strat in enumerate(strategies.mixinglist):
        deltaG += strategies.currentgenemixing[s] * deltaGtmp[:,:,s]
        
    if strategies.mixingstrategy == MixingStrategy.SELF_CONSISTENT:
        nstrat = len(strategies.mixinglist)
        deltaGMean = np.zeros([m, nstrat])
        totaldeltaGMean = np.zeros([nstrat])
        if not silent:
            print("--Gene strategy mixing")
        for s, strat in enumerate(strategies.mixinglist):
            deltaGMean[:,s] = np.mean(abs(deltaGtmp[:,:,s]), axis=0)
            totaldeltaGMean[s] = np.mean(deltaGMean[:,s])
            if not silent:
                print("Strategy = ", str(strat.gsstrategy), "; mean DeltaGs =", deltaGMean[:,s])
                print("Strategy = ", str(strat.gsstrategy), "; total mean =", totaldeltaGMean[s])
        newmixing = apply_deltas(strategies.currentgenemixing, m * totaldeltaGMean)    
        if not silent:
            print("current gene mixing =", strategies.currentgenemixing)
            print("new gene mixing =", newmixing)
        strategies._currentgenemixing = newmixing

    return deltaG

def compute_organism_deltas(orgindex, population, genefitness,
                            orgfitness, initialorgfitnessrange,
                            orgsimilarity, strategies):
    """Computes delta values stemming from the organism strategy.

    Args:
        orgindex (integer):
            The index i of the current organism.
        population (matrix of shape `(n, m)`):
            The full population matrix obtained by applying the
            gene variant fitness functions to the input data.
        genefitness (float vector of shape `(1, m)`):
            The list of gene fitness values.
        orgfitness (float vector of shape `(n, 1)`):
            The list of organism fitness values.
        initialorgfitness range (float):
            The initial length of the interval max(orgfitness)-min(orgfitness).
        orgsimilarity (float vector of shape `(n, 1)`):
            The kinship of the current organism to the others.
        strategies (pikaia.alg.Strategies):
            The evolutionary strategy to be applied.
    
    Returns:
        A vector shape `(0, m)` containing the changes to the gene
        fitness for the particular organism.
    """
    n = population.shape[0]
    m = population.shape[1]
    
    deltaO = np.zeros(m)
    if strategies.osstrategy == OSStrategy.BALANCED:
        for j in range(0, m):
            genecontribution = population[orgindex, j]*genefitness[j]
            if orgfitness[orgindex] == 0:
                deltaO[j] = 0
            else:
                # factor 2 (orgfitness)
                deltaO[j] = -2/n*(genecontribution/orgfitness[orgindex] - 1/m)*\
                                orgfitness[orgindex]
    elif strategies.osstrategy == OSStrategy.ALTRUISTIC: # experimental
        relatives = np.argsort(-orgsimilarity)
        nrel = min(n, strategies.kinrange)
        for i in range(0, nrel):
            rel = relatives[i]
            if rel == orgindex:
                continue
            
            for j in range(0, m):
                genecontribution = population[orgindex, j]*genefitness[j]
                if orgfitness[orgindex] == 0:
                    deltaO[j] = 0
                else:
                    deltaO[j] += -2/n*orgsimilarity[rel]*(genecontribution/orgfitness[orgindex] - 1/m)*\
                                (1/nrel)*(orgfitness[orgindex]-orgfitness[rel])/initialorgfitnessrange
                    
    elif strategies.osstrategy == OSStrategy.KIN_SELFISH:# experimental!
        relatives = np.argsort(-orgsimilarity)
        nrel = min(n, strategies.kinrange)
        for i in range(0, nrel):
            rel = relatives[i]
            if rel == orgindex:
                continue
            
            for j in range(0, m):
                genecontribution = population[orgindex, j]*genefitness[j]
                if orgfitness[orgindex] == 0:
                    deltaO[j] = 0
                else:
                    # factor 2 (orgfitness), 2 (orgsimilarity)
                    deltaO[j] += 2/n*(0.5-orgsimilarity[rel])*(genecontribution/orgfitness[orgindex] - 1/m)*\
                                (1/nrel)*(orgfitness[orgindex]-orgfitness[rel])/initialorgfitnessrange
                
    elif strategies.osstrategy == OSStrategy.SELFISH:
        relatives = np.argsort(-orgsimilarity)
        nrel = min(n, strategies.kinrange)
        orgfitnessrange = max(orgfitness)-min(orgfitness)
        for i in range(0, nrel):
            rel = relatives[i]
            if rel == orgindex:
                continue
           
            for j in range(0, m):
                genecontribution = population[orgindex, j]*genefitness[j]
                if orgfitness[orgindex] == 0:
                    deltaO[j] = 0
                else:
                    # factor 2 (orgsimilarity)
                    deltaO[j] += -2/n*orgsimilarity[rel]*(genecontribution/orgfitness[orgindex] - 1/m)*\
                                (1/nrel)*(orgfitness[orgindex]-orgfitness[rel])/initialorgfitnessrange
    elif strategies.osstrategy == OSStrategy.NONE:
        pass
    else:
        raise NameError("Unknown organism strategy:" + strategies.osstrategy)
    return deltaO

def mix_organism_deltas(deltaOtmp, genefitness, strategies, silent=False):
    """Mixes delta-contributions from multiple organism strategies.

    Note that if only one strategy is used, this function simply
    returns the input matrix as is.
    
    Args:
        deltaOtmp (matrix of shape `(n, m, nstrat)`):
            The delta-matrices for the to-be-mixed strategies.        
        strategies (pikaia.alg.Strategies):
            Contains the mixing strategy, either {fixed, self-consistent}.
    
    Returns:
        A matrix of shape `(n, m)` containing the mixed organism delta matrix.
    """ 
    n = deltaOtmp.shape[0]
    m = deltaOtmp.shape[1]
    
    deltaO = np.zeros([n,m])
    
    for s, strat in enumerate(strategies.mixinglist):
        deltaO += strategies.currentorgmixing[s] * deltaOtmp[:,:,s]
        
    if strategies.mixingstrategy == MixingStrategy.SELF_CONSISTENT:
        nstrat = len(strategies.mixinglist)
        deltaOMean = np.zeros([n, nstrat])
        totaldeltaOMean = np.zeros([nstrat])
        if not silent:
            print("--Organism strategy mixing")
        for s, strat in enumerate(strategies.mixinglist):
            deltaOMean[:,s] = np.dot(abs(deltaOtmp[:,:,s]), genefitness)
            totaldeltaOMean[s] = np.mean(deltaOMean[:,s])
            if not silent:
                print("Strategy = ", str(strat.osstrategy), "; mean DeltaOs =", deltaOMean[:,s])
                print("Strategy = ", str(strat.osstrategy), "; total mean =", totaldeltaOMean[s])
        newmixing = apply_deltas(strategies.currentorgmixing, totaldeltaOMean)    
        if not silent:
            print("current organism mixing =", strategies.currentorgmixing)
            print("new organism mixing =", newmixing)
        strategies._currentorgmixing = newmixing
    
    return deltaO
    
def iteration(population, genefitness, genesimilarity,
              orgsimilarity, initialorgfitnessrange,
              strategies, silent=False):
    """Runs a single, evolutionary step.

    One iteration takes a given population and gene fitness,
    calculates a new gene and organism fitness.

    Args:
        population (matrix of shape `(n, m)`):
            The population data in matrix form, rows are organisms,
            colums genes.
        genefitness (vector of shape `(1, m)`):
            The list of gene fitness values.
        genesimilarity (matrix of shape `(m, m)`):
            The symmetric gene kinship matrix.
        orgsimilarity (matrix of shape `(n, n)`):
            The symmetric organism kinship matrix.
        initialorgfitness range (float):
            The initial length of the interval max(orgfitness)-min(orgfitness).
        strategies (pikaia.alg.Strategies):
            Defines the used gene and organism strategy, respectively.
    
    Returns:
        Two vectors, first, the new gene fitness of shape `(1, m)`,
        second, the organism fitness values of shape `(n, 1)`
    """
    n = population.shape[0]
    m = population.shape[1]

    # get Delta contributions to gene fitness updates
    deltaGstmp = np.zeros([n,m,len(strategies.mixinglist)])
    deltaOstmp = np.zeros([n,m,len(strategies.mixinglist)])
    
    orgfitness = np.zeros([n,1])
    
    for i in range(0, n):
        orgfitness[i] = compute_organism_fitness(population[i,:], genefitness)

    for i in range(0, n):
        for s, strat in enumerate(strategies.mixinglist):
            deltaOstmp[i,:,s] = compute_organism_deltas(i, population, genefitness, 
                                                     orgfitness, initialorgfitnessrange,
                                                     orgsimilarity[i,:],
                                                     strat)
        for j in range(0, m):
            for s, strat in enumerate(strategies.mixinglist):
                deltaGstmp[i,j,s] = \
                    compute_gene_deltas(j, population[i,j], population[:,j],
                                        population[i,:], genefitness,
                                        genesimilarity[j,:], strat)
    
    # mix evolutionary strategies
    deltaGs = mix_gene_deltas(deltaGstmp, strategies, silent=silent)
    deltaOs = mix_organism_deltas(deltaOstmp, genefitness, strategies, silent=silent)
    if not silent:  
        print("deltaGs = ", deltaGs)
        print("deltaOs = ", deltaOs)
    deltas = deltaGs + deltaOs
    deltaG = np.sum(deltaGs,axis=0)
    deltaO = np.sum(deltaOs,axis=0)
    delta = np.sum(deltas,axis=0)
    if not silent:
        print("deltaG = ", deltaG)
        print("deltaO = ", deltaO)
        print("delta = ", delta)

    # apply replicator equations
    newgenefitness = np.zeros(m)
    for j in range(0, m):
        newgenefitness[j] = genefitness[j]*(1+delta[j])
    sumfitness = sum(newgenefitness)
    for j in range(0, m):
        newgenefitness[j] /= sumfitness
    neworganismfitness = compute_all_organism_fitness(population, newgenefitness)
    
    return newgenefitness, neworganismfitness

def compute_gene_similarity(population):
    """Computes the similarity/kinship matrix for genes.

    Args:
        population (matrix of shape `(n, m)`):
            The population data in matrix form, rows are organisms,
            colums genes.
    
    Returns:
        the similarity matrix `(m, m)`,
    """
    n = population.shape[0]
    m = population.shape[1]
    genediversity = np.zeros([m,m])
    for j in range(0,m):
        for l in range(0,m):
            tmp = np.linalg.norm(population[:,j] - population[:,l])
            genediversity[j,l] += tmp
                
    genesimilarity = 1 - genediversity/float(n)
    print("Gene similarity = ")
    print(genesimilarity)
    return genesimilarity

def compute_organism_similarity(population):
    """Computes the similarity/kinship matrix for organisms.

    Args:
        population (matrix of shape `(n, m)`):
            The population data in matrix form, rows are organisms,
            colums genes.
    
    Returns:
        the similarity matrix `(n, n)`,
    """
    n = population.shape[0]
    m = population.shape[1]
    orgdiversity = np.zeros([n,n])
    for i in range(0, n):
        for l in range(0, n):
            tmp = np.linalg.norm(population[i,:] - population[l,:])
            orgdiversity[i,l] += tmp
                
    orgsimilarity = 1 - orgdiversity/float(m)
    print("Organism similarity = ")
    print(orgsimilarity)
    return orgsimilarity





