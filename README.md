# pikaia - Genetic AI

pikaia is the Python implementation of Genetic AI (evolutionary simulation for data analysis).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pikaia 

```bash
pip install pikaia
```

## Usage 

We provide here the code for the "hello_model" example

```python
import pikaia
import pikaia.alg
          

rawdata = np.zeros([3,3])
rawdata[:,:] = [[ 300, 10, 2],
                [ 600,  5, 2],
                [1500,  4, 1]]
# defines the variant fitness rules
gvfitnessrules = ["inv_percentage", "inv_percentage", "inv_percentage"]
# converts the raw data to a genetic population
data = pikaia.alg.Population(rawdata, gvfitnessrules)
# defines the used evolutionary strategies
strategy = ["GS Dominant", "OS Balanced"]
iterations = 1

# creating the genetic model
model = pikaia.alg.Model(data, strategy)

initialgenefitness = [1.0/3.0, 1.0/3.0, 1.0/3.0]
# returns the gene fitness values after 1 iteration
model.complete_run(initialgenefitness, iterations)

```

## Examples
```python
# provides the data for a small decision problem
example3x3 = pikaia.examples.assemble_example("3x3-DomBal+AltSal")

# provides the data for a real-world decision problem
example10x5 = pikaia.examples.assemble_example("10x5-DomBal+AltSal")

# use genetic ai to search a datafile using keywords and rank results
# for a more detailed example we refer to examples/geneticAI_run_search_example.py
search = pikaia.search.Search(data, orgs_labels, gens_labels)
fitnessOrganisms, fitnessGenes = search.search_request(query, top_k=5)
```

For details see examples/README.md.

## Scientific Background

Please find the preprint of Genetic AI [here](http://arxiv.org/abs/2501.19113)


In Genetic AI, we convert a data problem to a model of genes and organisms. Afterwards, we run evolutionary simulations to obtain understanding of the input data.

Genetic AI is an AI that does not use training data to 'learn' but fully autonomously analyzes a problem. This is done by evolutionary strategies that cover certain 'behavior' and correlations of the input data.

## License

[MIT](https://choosealicense.com/licenses/mit/)
