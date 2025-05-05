# Example Scripts

We provide three example scripts that show how Genetic AI can be used for data analysis and search result ranking.

If you already have installed the pikaia package, you can run the script from the terminal
using the following command (executed from inside the examples folder):

`$ python geneticAI_hello_model.py`

If you do not have the pikaia package installed, you have to add src/pikaia to your PYTHONPATH 
in order to be able to import the modules properly. Make sure you have all the necessary
packages installed. If you're using pip, run `pip install -r requirements.txt` in the top-level
folder. Then you can run our example scripts. Here's how to run the hello_model
script from the top-level folder:

`$ PYTHONPATH=src python examples/geneticAI_hello_model.py`

## Script 1: Hello World
Runs a small evolutionary simulation on three data sets and three genes.

## Script 2: Arxiv Examples
Runs some examples from the [arxiv paper](https://arxiv.org/abs/2501.19113) and plots the results:

`$ python geneticAI_run_arxiv_examples.py`

## Script 3: Self-consistent Examples
Runs the self-consistent examples from the [arxiv paper](https://arxiv.org/abs/2501.19113) and plots the results:

`$ python geneticAI_run_selfconsistency_example.py`


## Script 4: Search Example
Runs a simple text-based search on a movie dataset. You can either run the script
with the default arguments:

`$ geneticAI_run_search_example.py`

or provide an optional query (a string of comma-separated keywords used for the
search) or an optional path to a custom data matrix:

`$ python geneticAI_run_search_example.py --query "romance, movie, film noir" --data-path path/to/data`

The data matrix should be a csv-file with each row being a movie and each column being a
keyword that describes the movie features. The "title" column will be used as the movie
title and it is assumed that there is no feature with the name "title". The year column
will be interpreted as the movie's release year. All other columns describe content
keywords and can have values ranging from 0 to 1.

```
    title,year,adventure,interplanetary politics
    E.T. the Extra-Terrestrial,1982,1.0,0.705667
    Lawrence of Arabia,1962,1.0,0.0
    What Ever Happened to Baby Jane?,1962,0.0,0.0
    American Vandal,2017,0.0,0.0
```

# Defining Strategies
In Genetic AI, evolutionary strategies determine the dynamics of the evoluationary game.
They describe how genes and organisms behave during the evolutionary simulation
and capture symmetries and correlations in the data.

Currently, the following gene strategies are implemented in pikaia:
- dominant
- altruistic
- selfish
- kin_altruistic

The following organism strategies are available:
- balanced
- altruistic
- selfish
- kin_selfish

Pikaia lets you pre-define the gene and organis strategies to use in the simulations.
First, you need to instantiate a `Strategies` object with the gene strategy
(`GSStrategy`) and the organism strategy (`OSStrategy`). Then, you create your
pikaia model with the population (data) and the strategies:

```
strategies = pikaia.alg.Strategies(GSStrategy.DOMINANT, OSStrategy.BALANCED)
model = pikaia.alg.Model(population, strategies)
```

You can also mix strategies:

``` 
strat1 = pikaia.alg.Strategies(GSStrategy.DOMINANT, OSStrategy.BALANCED)
strat2 = pikaia.alg.Strategies(GSStrategy.ALTRUISTIC, OSStrategy.SELFISH, kinrange=10)

strategies = pikaia.alg.Strategies(GSStrategy.MIXED, OSStrategy.MIXED,
                                   kinrange=None,
                                   mixingstrategy=MixingStrategy.FIXED,
                                   mixinglist=[strat1,strat2],
                                   initialgenemixing=[0.5, 0.5],
                                   initialorgmixing=[0.5, 0.5])

```

In the example above, we first define two combinations of strategies (`strat1`, `strat2`).
When instantiating a `Strategies` object, we pass them as `mixinglist`, indicating that
pikaia can mix the gene strategy `DOMINANT` with `ALTRUISTIC` and organism strategy
`BALANCED` with `SELFISH`. By setting `mixingstrategy=MixingStrategy.FIXED`, we define
that the mixing ratio for the strategies will not be changed in the simulation.

Pikaia can also find the best mixing ratios for the strategies with respect to your data.
You can do this by setting `mixingstrategy=MixingStrategy.SELF_CONSISTENT`:

```
strategies = pikaia.alg.Strategies(GSStrategy.MIXED, OSStrategy.MIXED,
                                   kinrange=None,
                                   mixingstrategy=MixingStrategy.SELF_CONSISTENT,
                                   mixinglist=[strat1,strat2],
                                   initialgenemixing=[0.5, 0.5],
                                   initialorgmixing=[0.5, 0.5])
```
In this example, pikaia will optimize the gene mixing and organism mixing
ratios depending on your data in the evolutionary simulation.
You can obtain the final mixing ratios from the `strategies` object after
the simulation (i.e. after calling `model.complete_run()`):

``` 
initialgenefitness = [1 / n_genes] * n_genes # uniform initial gene fitness
model.complete_run(initialgenefitness, iterations=100)

print(strategies.currentgenemixing) # ratio of gene strategies [GSStrategy.DOMINANT, GSStrategy.ALTRUISTIC]
print(strategies.currentorgmixing) # ratio of organism strategies [OSStrategy.BALANCED, OSStrategy.SELFISH]
``` 
