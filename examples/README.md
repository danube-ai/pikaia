# Example Scripts

We provide three example scripts that show how Genetic AI can be used for data analysis and search result ranking.

If you already have installed the pikaia package, you can run the script from the terminal
using the following command (executed from inside the examples folder):

`$ python geneticAI_hello_model.py`

If you do not have the pikaia package installed, you have to add src/pikaia to your PYTHONPATH 
in order to be able to import the modules properly. Here's an example for running the hello_model
script from the top-level folder:

`$ PYTHONPATH=src python examples/geneticAI_hello_model.py`

## Script 1: Hello World
Runs a small evolutionary simulation on three data sets and three genes.

## Script 2: Arxiv Examples
Runs the examples from the [arxiv paper](https://arxiv.org/abs/2501.19113) and plots the results.

## Script 3: Search Example
Runs a simple text-based search on a movie dataset. You can either run the script
with the default arguments:

`$ python geneticAI_run_search_example.py`

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
