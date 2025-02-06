import numpy as np

class Model:
    def __init__(self, inputdata, strategies, 
                 genelabels=None, orgslabels=None,
                 linestyles=None, markerstyles=None):
        self._data = inputdata
        self._strategies = strategies
        self._genelabels = genelabels
        self._orgslabels = orgslabels
        self._linestyles = linestyles
        self._markerstyles = markerstyles

    @property
    def data(self):
        return self._data
        

    def complete_run(self, initialgenefitness, maxiteration=1, epsilon=None, silent=False):
        """Runs an evolutionary simulation.

        Runs a evolutionary simulation until either maxiter 
        is reached or the difference of gene fitness values
        from two consecutive iterations are below epsilon.

        Args:
            initialgenefitness (vector of shape `(1, m)`):
                The initial fitness used in this simulation.
        
        Returns:
            the similarity matrix `(n, n)`,
        """
        self._maxiter = maxiteration
        self._epsilon = epsilon
        genefitness = initialgenefitness
        print("initial gene fitness = ", genefitness)
        initialorganismfitness = compute_all_organism_fitness(self.data.population, genefitness)
        if not silent:
            print("initial organism fitness = ", initialorganismfitness)
        
        #iter = self._maxiter
        n = self.data.population.shape[0]
        m = self.data.population.shape[1]
        # helper structures storing iterations
        self._gps = np.zeros([self._maxiter+1, m])
        self._ofs = np.zeros([self._maxiter+1, n])
        self._gps[0,:] = genefitness
        self._ofs[0,:] = initialorganismfitness

        # compute similarities necessary for some strategies
        genesimilarity = compute_gene_similarity(self.data.population)
        orgsimilarity = compute_organism_similarity(self.data.population)
        
        for k in range(0, self._maxiter):
            if not silent:
                print("+++ Iteration ", k," +++")
            genefitness, fitness = iteration(self.data.population, genefitness,
                                            genesimilarity, orgsimilarity,
                                            self._strategies, silent=silent)
            self._gps[k+1,:] = genefitness
            self._ofs[k+1,:] = fitness
            if not silent:
                print("gene fitness = ", genefitness)
                print("organism fitness = ", fitness)
            delta = np.linalg.norm(self._gps[k+1,:]-self._gps[k,:])
            self._iterESE = k + 1
            if self._epsilon is not None and delta < self._epsilon:
                print("reached ESE after ", self._iterESE, "iterations. Final delta = ", delta)
                break
        return genefitness, fitness

class Population:
    def __init__(self, rawdata, gvfitnessrule):
        self._rawdata = rawdata
        self._n = self._rawdata.shape[0] #number of organisms
        self._m = self._rawdata.shape[1] #number of genes
        self._gvfitnessrules = gvfitnessrule
        self._population = np.zeros([self._n,self._m]) 
        for j in range(0, self._m):
            # the populations derive from the raw data by applying gene variant 
            # fitness functions
            self._population[:,j] = compute_gene_variant_fitness(self._rawdata[:,j], 
                                                                 self._gvfitnessrules[j])
        print("Population = ", self._population)

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
    def population(self):
        return self._population
    
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
        gvrule:
            The gene variant fitness function applied.

    Returns:
        A float, the gene variant fitness.
    """
    if "inv" in gvfrule:
        invert = True
    else:
        invert = False
    
    if "cap" in gvfrule:
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
                        genesimilarity, strategy="GS Dominant"):
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
        strategy (string):
            The evolutionary strategy to be applied.
         
    Returns:
        A float, the Delta(i,j) value for a particular gene 
        and organism, respectively.
    """    
    n = len(gene)
    m = len(organism)
    if strategy == "GS Dominant":
        deltaG = 1/float(n)*genefitness[geneindex]*(genevariantfitness-1/2)*2
    elif strategy == "GS Selfish":
        deltaG = 0
        for j in range(0, m):
            if j == geneindex:
                continue
            deltaG += -1/float(n)*genesimilarity[j]*genefitness[geneindex]*\
                                  (genevariantfitness-1/2)*(organism[j] - genevariantfitness)
        deltaG  = deltaG/float(m)
    elif strategy == "GS Kin-Altruistic":
        deltaG = 0
        for j in range(0, m):
            if j == geneindex:
                continue
            deltaG += 1/float(n)*(0.5-genesimilarity[j])*genefitness[geneindex]*\
                                (genevariantfitness-1/2)*(organism[j] - genevariantfitness)
        deltaG  = deltaG/float(m)
    elif strategy == "GS Altruistic":
        deltaG = 0
        for j in range(0, m):
            if j == geneindex:
                continue
            deltaG += 1/float(n)*genesimilarity[j]*genefitness[geneindex]*(genevariantfitness-1/2)*\
                                genefitness[j]*(organism[j] - genevariantfitness)
        deltaG  = deltaG/float(m)
    elif strategy == "GS None":
        deltaG = 0
    else:
        raise NameError("Unknown gene strategy:" + strategy)
    
    return deltaG

def compute_organism_deltas(orgindex, population, genefitness,
                            orgfitness, orgsimilarity, strategy="OS Balance"):
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
        orgsimilarity (float vector of shape `(n, 1)`):
            The kinship of the current organism to the others.
        strategy (string):
            The evolutionary strategy to be applied.
    
    Returns:
        A vector shape `(0, m)` containing the changes to the gene
        fitness for the particular organism.
    """
    n = population.shape[0]
    m = population.shape[1]
    
    deltaO = np.zeros(m)
    if strategy == "OS Balanced":
        for j in range(0, m):
            genecontribution = population[orgindex, j]*genefitness[j]
            if orgfitness[orgindex] == 0:
                deltaO[j] = 0
            else:
                deltaO[j] = -1/n*(genecontribution/orgfitness[orgindex] - 1/m)*orgfitness[orgindex]
    elif strategy == "OS Altruistic":
        for i in range(0, n):
            if i == orgindex:
                continue
            
            for j in range(0, m):
                genecontribution = population[orgindex, j]*genefitness[j]
                if orgfitness[orgindex] == 0:
                    deltaO[j] = 0
                else:
                    deltaO[j] += -1/n*orgsimilarity[i]*(genecontribution/orgfitness[orgindex] - 1/m)*\
                                (1/n)*(orgfitness[orgindex]-orgfitness[i])
    elif strategy == "OS Kin-Selfish":
        for i in range(0, n):
            if i == orgindex:
                continue
            
            for j in range(0, m):
                genecontribution = population[orgindex, j]*genefitness[j]
                if orgfitness[orgindex] == 0:
                    deltaO[j] = 0
                else:
                    deltaO[j] += 1/n*(0.5-orgsimilarity[i])*(genecontribution/orgfitness[orgindex] - 1/m)*\
                                (1/n)*(orgfitness[orgindex]-orgfitness[i])
                
    elif strategy == "OS Selfish":
        for i in range(0, n):
            if i == orgindex:
                continue
           
            for j in range(0, m):
                genecontribution = population[orgindex, j]*genefitness[j]
                if orgfitness[orgindex] == 0:
                    deltaO[j] = 0
                else:
                    deltaO[j] += -1/n*orgsimilarity[i]*(genecontribution/orgfitness[orgindex] - 1/m)*\
                                (1/n)*(orgfitness[orgindex]-orgfitness[i])
    elif strategy == "OS None":
        pass
    else:
        raise NameError("Unknown organism strategy:" + strategy)
    return deltaO

def iteration(population, genefitness, genesimilarity,
              orgsimilarity, strategy, silent=False):
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
        strategy (string vector of length 2):
            Defines the used gene and organism strategy, respectively.
    
    Returns:
        Two vectors, first, the new gene fitness of shape `(1, m)`,
        second, the organism fitness values of shape `(n, 1)`
    """
    n = population.shape[0]
    m = population.shape[1]

    # get Delta contributions to gene fitness updates
    deltaGs = np.zeros([n,m])
    deltaOs = np.zeros([n,m])
    orgfitness = np.zeros([n,1])
    for i in range(0, n):
        orgfitness[i] = compute_organism_fitness(population[i,:], genefitness)

    for i in range(0, n):
        deltaOs[i,:] = compute_organism_deltas(i, population, genefitness, orgfitness, orgsimilarity[i,:], strategy[1])
        for j in range(0, m):
            deltaGs[i,j] = compute_gene_deltas(j, population[i,j], population[:,j], population[i,:], genefitness,
                                               genesimilarity[j,:], strategy[0])
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





