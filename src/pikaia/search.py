import datetime

import numpy as np

import pikaia
import pikaia.alg

from pikaia.alg import GSStrategy
from pikaia.alg import OSStrategy

class Search:
    """
    Class that performs a string-based search over a data matrix
    and sorts the results using Genetic AI.

    Args:
        inputdata(matrix of shape `(n, m)`): A matrix containing
            structured data.
        orgs_labels(list of strings, optional): The organism labels used 
            for plotting.
        gene_labels(list of strings, optional): The gene labels used
            for plotting. Length should be m.
        
    """
    def __init__(self, input_data, orgs_labels, gene_labels):
        self._data = input_data
        self._gene_labels = gene_labels
        self._orgs_labels = orgs_labels

    def search_request(self, query: str, top_k: int = None, silent=False) -> list:
        """
        Performs a search request on the data given a query string.

        Args:
            query (str): a string containing comma-separated keywords.
            top_k (int): how many top-ranked organisms to return.
        
        Returns:
            Tuple[list, list] containing the organism and gene fitness
            values at the end of the simulation.
        """
        # Process query
        split_query = [f.strip().lower() for f in query.split(",")]
        query_features = [f for f in split_query if f in self._gene_labels]
        query_feature_ids = [self._gene_labels.index(f) for f in query_features]
        query_year = next((int(f) for f in split_query if is_year(f)), 0)

        # For testing purposes set row cutoff
        cutoff = None
        if cutoff is None:
            cutoff = self._data.shape[0]

        # Get the relevant column for each feature and create a new matrix
        data_subset = self._data[:cutoff, query_feature_ids]

        # Use percentage rule for all features except year
        gvfitnessrules = ["percentage"] * len(query_features)

        # Get relevant columns from matrix
        if query_year > 0:
            # Convert year column -> calculate the distance to the year in the query
            year_col = abs(self._data[:cutoff, self._gene_labels.index("year")] - query_year)

            # Add to the data_subset
            data_subset = np.column_stack((data_subset, year_col))

            # Special handling for year
            gvfitnessrules += ["inv_percentage"]

        # Print logs
        print("Selected features:", query_features, f"(ids: {query_feature_ids})")
        print("Number of non-zero values per selected feature:")
        for f_id, f in zip(query_feature_ids, query_features):
            print(f" - {f}:", len([w for w in self._data[:, f_id] if w != 0]))

        print(f"Subset of data matrix (shape: {data_subset.shape})\n", data_subset)

        # Convert raw data to population
        population = pikaia.alg.Population(data_subset, gvfitnessrules)

        strategies = pikaia.alg.Strategies(GSStrategy.DOMINANT, OSStrategy.BALANCED)

        # Use this for AltSel strategies
        #strategies = pikaia.alg.Strategies(GSStrategy.ALTRUISTIC, OSStrategy.SELFISH,
        #kinrange=10)

        # Initialize model
        model = pikaia.alg.Model(population, strategies)

        # Start with a uniform distribution
        n_features = data_subset.shape[1]
        initialgenefitness = [1 / n_features] * n_features

        # Run simulation
        e = 0.00005
        gene_fitness, fitness = model.complete_run(initialgenefitness, maxiteration=100,
                                                   epsilon=e, silent=silent)

        # Function to get all non-zero features for a selected organism
        def get_features(organism):
            org_index = self._orgs_labels.index(organism)
            return [f_name for f_weight, f_name in zip(self._data[org_index,:], self._gene_labels)
                    if f_weight != 0]

        # Function to get the feature values for a selected organism
        def get_feature_values(organism):
            org_index = self._orgs_labels.index(organism)
            return [(f_name, f_weight)
                    for f_weight, f_name in zip(data_subset[org_index,:], query_features)]

        # Sort organisms with their corresponding fitness value
        org_fitness = list(sorted(
            [(org_label, fit, get_feature_values(org_label))
             for org_label, fit in zip(self._orgs_labels, fitness)],
            key=lambda x: x[1],
            reverse=True
        ))

        gene_fitness_labels = [(l, f) for l, f in zip(query_features, gene_fitness)]

        # Return the top_k fittest organisms
        return org_fitness[:top_k], gene_fitness_labels

def is_year(s):
    """Checks if a string is a year number."""
    try:
        year = int(s)
    except ValueError:
        return False
    return year >= 1890 and year <= datetime.date.today().year + 1
