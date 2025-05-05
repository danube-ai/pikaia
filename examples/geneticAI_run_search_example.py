"""
Script that can be used to perform a Genetic-AI-powered search on
example movie data. Can be called like: 
    $ python geneticAI_run_search_example.py

    Or with optional arguments:
    $ python geneticAI_run_search_example.py --query "romance, movie, film noir" --data-path path/to/data
"""

import argparse
import os
from typing import Union, Tuple

import numpy as np
import pandas as pd

from pikaia.search import Search


def load_data(path: Union[str, os.PathLike]) -> Tuple[np.array, list, list]:
    """
    Loads the data and converts it to numpy.

    Args:
        path: the path to a csv-file containing organisms (rows) and
          genes (features/tags/content descriptors). All non-title
          columns must be numerical. Example content of the file:

            title,year,adventure,interplanetary politics
            E.T. the Extra-Terrestrial,1982,1.0,0.705667
            Lawrence of Arabia,1962,1.0,0.0
            What Ever Happened to Baby Jane?,1962,0.0,0.0
            American Vandal,2017,0.0,0.0

    Returns:
        Tuple consisting of the data matrix, the organism labels
        and the gene lables.
    """
    data = pd.read_csv(path)

    return data.to_numpy()[:,1:], list(data["title"]), list(data.columns)[1:]

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.abspath(__file__))
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="heist, movie, dream",
                        help="Comma-separated string of keywords.")
    parser.add_argument("--data-path", type=str,
                        default=f"{file_dir}/data/movie_matrix.csv",
                        help="Path to a data-matrix in correct format.")
    args = parser.parse_args()

    # Load data
    np_data, orgs_labels, gens_labels = load_data(args.data_path)

    # Perform search
    search = Search(np_data, orgs_labels, gens_labels)
    print("Running evolutionary simulation...")
    organisms, genes = search.search_request(args.query, top_k=5, silent=True)

    print("\nFinished simulation.")
    print(f"\nTop results for '{args.query}':")
    max_title_len = 30
    for movie, fitness, features in organisms:
        truncated_title = (movie[:max_title_len-3] + "...") \
            if len(movie) > max_title_len else movie.ljust(max_title_len)
        print(f" - {truncated_title} {fitness:.4f} {str(features)}")

    print("\nGene fitness values:")
    for gene_name, fitness in genes:
        print(f" - {gene_name:<30} {fitness:.4f}")
