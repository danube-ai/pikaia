import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import string

import pikaia.alg

def initialize_plotting_variables():
    """Initialize styles and markers for plotting.

    
    Returns:
        linestyles (list of linestyle specifiers).
        two sets of markers (list of strings).
    """
    
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as ':'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'
    linestyle_tuple = [
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 5))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),
     ('loosely dashdotted',    (0, (3, 10, 1, 10))), 
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
    linestyles = []
    for i, (name, linestyle) in enumerate(linestyle_str):
        linestyles.append(linestyle)
    for i, (name, linestyle) in enumerate(linestyle_tuple):
        linestyles.append(linestyle)
    markerstylesSet1 = ["o","^","s", ">", "p", "X", ".", "*", "1", "3"]
    markerstylesSet2 = ["x","v", "D","<", "P", "h", "8", "d", "2", "4"]
    return linestyles, markerstylesSet1, markerstylesSet2

def plot_gene_fitness(modellist, figurenr, show=True, savename=None,
                      fontsize=12, linewidth=3, postfix=None):
    """Plots the gene fitness values over iterations.

    Args:
        modellist (List of geneticai models):
            List of geneticai models to be plotted.
        figurenr (integer):
            The figure window specifier.
        show (boolean):
            Whether the finished plots are shown.
        savename (String):
            Whether the finished plots are saved as png and pdf.
        fontsize (integer):
            Fontsizes in the plots
        linewidth (integer):
            Linewidths in the plots
           

    """
    
    plt.figure(figurenr)
    plt.rcParams.update({'font.size': fontsize})
    
    maxpiter = 0
    for model in modellist:
        if model._iterESE > maxpiter:
            maxpiter = model._iterESE
    
    if maxpiter > 20:
        inc = round(maxpiter/10)
        plt.xticks(range(0,maxpiter, inc)) 
    else:
        inc = 5
        plt.xticks(range(0,maxpiter)) 

    if postfix is None:
        postfix = []
        for model in modellist:
            postfix.append("")

    for model, pf in zip(modellist, postfix):
        n = model._gps.shape[1]
        for j in range(0, n):
            plt.plot(range(0, model._iterESE+1), model._gps[:model._iterESE+1,j],
                    label=model._genelabels[j]+pf, linestyle=model._linestyles[j],
                    marker=model._markerstyles[j], markersize=8, markevery=inc,
                    lw=linewidth)
        plt.ylabel('gene fitness [%]')
        plt.xlabel('iterations')
        
    
    plt.xlim(0,maxpiter)
    plt.legend(handlelength=5, fontsize=10)
    if savename is not None:
        plt.savefig(savename + ".pdf", format="pdf", bbox_inches="tight")
        plt.savefig(savename + ".png", format="png", bbox_inches="tight")
    if show:
        plt.show()

def plot_organism_fitness(modellist, figurenr, maxitershown, show=True, savename=None,
                          fontsize=12, linewidth=3):
    
    """Plots the organism fitness values over iterations.

    Args:
        modellist (List of geneticai models):
            List of geneticai models to be plotted.
        figurenr (integer):
            The figure window specifier.
        maxitershown (integer):
            The maximum iteration shown in the plot.
        show (boolean):
            Whether the finished plots are shown.
        savename (String):
            Whether the finished plots are saved as png and pdf.
        fontsize (integer):
            Fontsizes in the plots
        linewidth (integer):
            Linewidths in the plots
           

    """
    plt.figure(figurenr)
    plt.rcParams.update({'font.size': fontsize})
    
    
    maxpiter = 0
    for model in modellist:
        if model._iterESE > maxpiter:
            maxpiter = model._iterESE
    
    if maxitershown is not None and maxitershown < maxpiter:
        maxpiter = maxitershown
    
    if maxpiter > 20:
        inc = round(maxpiter/10)
        plt.xticks(range(0,maxpiter, inc)) 
    else:
        inc = 5
        plt.xticks(range(0,maxpiter)) 

    
    for model in modellist:
        n = model._ofs.shape[1]
        for j in range(0, n):
            plt.plot(range(0, model._iterESE+1), model._ofs[:model._iterESE+1,j],
                    label=model._orgslabels[j], linestyle=model._linestyles[j],
                    marker=model._markerstyles[j], markersize=8, markevery=inc,
                    lw=linewidth)
        plt.ylabel('organism fitness [%]')
        plt.xlabel('iterations')
        
    plt.xlim(0, maxpiter)
    plt.legend(handlelength=5, fontsize=10)
    if savename is not None:
        plt.savefig(savename + ".pdf", format="pdf", bbox_inches="tight")
        plt.savefig(savename + ".png", format="png", bbox_inches="tight")
    if show:
        plt.show()

def plot_mixing(modellist, figurenr, maxitershown, show=True, savename=None,
                fontsize=12, linewidth=3, postfix=None):
    
    """Plots the mixing values over iterations.

    Args:
        modellist (List of geneticai models):
            List of geneticai models to be plotted.
        figurenr (integer):
            The figure window specifier.
        maxitershown (integer):
            The maximum iteration shown in the plot.
        show (boolean):
            Whether the finished plots are shown.
        savename (String):
            Whether the finished plots are saved as png and pdf.
        fontsize (integer):
            Fontsizes in the plots
        linewidth (integer):
            Linewidths in the plots
           

    """
    plt.figure(figurenr)
    plt.rcParams.update({'font.size': fontsize})
    
    
    maxpiter = 0
    for model in modellist:
        if model._iterESE > maxpiter:
            maxpiter = model._iterESE
    
    if maxitershown is not None and maxitershown < maxpiter:
        maxpiter = maxitershown
    
    if maxpiter > 20:
        inc = round(maxpiter/10)
        plt.xticks(range(0,maxpiter, inc)) 
    else:
        inc = 5
        plt.xticks(range(0,maxpiter)) 

    if postfix is None:
        postfix = []
        for model in modellist:
            postfix.append("")
    
    for model, p in zip(modellist,range(0, len(postfix))):
        nstrat = model._genemixing.shape[1]
        for s in range(0, nstrat):
            plt.plot(range(0, model._iterESE+1), model._genemixing[:model._iterESE+1,s],
                    label=str(model.strategies.mixinglist[s].gsstrategy)+postfix[p],
                    linestyle=model._linestyles[2*s], marker=model._markerstyles[2*p+s],
                    markersize=8, markevery=inc, lw=linewidth)
            plt.plot(range(0, model._iterESE+1), model._orgmixing[:model._iterESE+1,s],
                    label=str(model.strategies.mixinglist[s].osstrategy)+postfix[p],
                    linestyle=model._linestyles[2*s+1], marker=model._markerstyles[2*p+s],
                    markersize=8, markevery=inc, lw=linewidth)
        plt.ylabel('mixing [%]')
        plt.xlabel('iterations')
        
    plt.xlim(0, maxpiter)
    plt.legend(handlelength=5, fontsize=10)
    if savename is not None:
        plt.savefig(savename + ".pdf", format="pdf", bbox_inches="tight")
        plt.savefig(savename + ".png", format="png", bbox_inches="tight")
    if show:
        plt.show()