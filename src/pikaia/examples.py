import numpy as np
import string

import pikaia.alg
from pikaia.alg import GVRule

rawdata3x3 = np.zeros([3,3])
rawdata3x3B = np.zeros([3,3])
rawdata3x3[:,:] = [[ 300, 10, 2],
                   [ 600,  5, 2],
                   [1500,  4, 1]]
rawdata3x3B[:,:] = [[ 300, 10, 2],
                    [ 600,  5, 1],
                    [700,  4, 1]]

rawdata10x5 = np.zeros([10,5])
rawdata10x5[:,:] = [[300, 10 , 2, 0, 2.5],
                    [600, 5, 2, 1, 3.0], 
                    [1500, 4, 1, 2, 4.0],
                    [400, 8, 2, 0, 3.5],
                    [500, 8, 2, 1, 3.0],
                    [700, 5, 2, 1, 4.5],
                    [900, 6, 1, 1, 4.0],
                    [1100, 6, 1, 2, 3.5],
                    [1300, 5, 2, 2, 5.0],
                    [1700, 4, 1, 2, 5.0]]

class Example:
    """Class encapsulating the raw data and plotting information.

    Args:
        inputdata(matrix of shape `(n, m)`): A matrix containing
            structured data.
        genelabels(list of strings, optional): The gene labels used 
            for plotting. Length should be m.
        orglabels(list of strings, optional): The organism labels used 
            for plotting.
        labelpostfix(list of strings, optional): Postfixes describing
            the strategy. Will be added to the labels when plotting.
    """
    def __init__(self, inputdata, genelabels=None, orgslabels=None, labelpostfix=None):
        self._data = inputdata  
        self._genelabelsbase = genelabels
        if self._genelabelsbase is None:
            self._genelabels = self._genelabelsbase
        else:
            self._genelabels = []
            for k, pf in enumerate(labelpostfix):
                # import pdb; pdb.set_trace()
                self._genelabels.append([])
                for j in range(0, self._data.m):
                    self._genelabels[k].append(self._genelabelsbase[j] + pf)

        self._orgslabelsbase = orgslabels
        if self._orgslabelsbase is None:
            self._orgslabels = self._orgslabelsbase
        else:    
            self._orgslabels = []
            for k, pf in enumerate(labelpostfix):
                self._orgslabels.append([])
                for i in range(0, self._data.n):
                    self._orgslabels[k].append(self._orgslabelsbase[i] + pf)
                        
    @property
    def exampledata(self):
        return self._data

    def get_gene_labels(self, set=0):
        return self._genelabels[set]
    
    def get_org_labels(self, set=0):
        return self._orgslabels[set]


def assemble_example(specifier):
    """Returns a setup ready for modelling.

    Returns a datastructure for Genetic Ai including
    model and ploting parameters.

    Args:
        specifier (string):
        Giving '3x3-DomBal+AltSal' returns a small example
        with n=3 and m=3;
        Giving '10x5-DomBal+AltSal' returns a larger example
        with n=10 and m=5.
            
    
    Returns:
        an Example data structure.
    """
    if specifier == "3x3-DomBal+AltSal":
        
        gvfitnessrules = [GVRule.PERCENTAGE_INVERTED,
                          GVRule.PERCENTAGE_INVERTED,
                          GVRule.PERCENTAGE_INVERTED]
        inputdata = pikaia.alg.Population(rawdata3x3, gvfitnessrules)

        # defining plotting labels
        genelabels = ["gene 1 = price", "gene 2 = time", "gene 3 = stops"]
        orgslabels = []
        for i in range(0, inputdata.n):
            orgslabels.append("flight " + string.ascii_uppercase[i])
        labelpostfixes = ["(DomBal)", "(AltSel)","(sc)"]

        return Example(inputdata, genelabels, orgslabels, labelpostfixes)

    elif specifier == "3x3-DomBal+AltSal-Capped":
        
        gvfitnessrules = [GVRule.PERCENTAGE_INVERTED_CAPPED,
                          GVRule.PERCENTAGE_INVERTED_CAPPED,
                          GVRule.PERCENTAGE_INVERTED_CAPPED]
        
        inputdata = pikaia.alg.Population(rawdata3x3, gvfitnessrules)

        # defining plotting labels
        genelabels = ["gene 1 = price", "gene 2 = time", "gene 3 = stops"]
        orgslabels = []
        for i in range(0, inputdata.n):
            orgslabels.append("flight " + string.ascii_uppercase[i])
        labelpostfixes = ["(DomBal)", "(AltSel)","(sc)"]

        return Example(inputdata, genelabels, orgslabels, labelpostfixes)
    
    elif specifier == "10x5-DomBal+AltSal":
        
        gvfitnessrules = [GVRule.PERCENTAGE_INVERTED,
                          GVRule.PERCENTAGE_INVERTED,
                          GVRule.PERCENTAGE_INVERTED,
                          GVRule.PERCENTAGE,
                          GVRule.PERCENTAGE] 
        inputdata = pikaia.alg.Population(rawdata10x5, gvfitnessrules)
        genelabels = ["gene 1 = price", "gene 2 = time", "gene 3 = stops",
                      "gene 4 = luggage", "gene 5 = rating"]
        orgslabels = []
        for i in range(0, inputdata.n):
            orgslabels.append("flight " + string.ascii_uppercase[i])
        labelpostfixes = ["(DomBal)", "(AltSel)", "(sc)"]

        return Example(inputdata, genelabels, orgslabels, labelpostfixes)
        
            
    else:
        raise ValueError("Unknown example specifier")