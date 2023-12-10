# CLAM: Clustered Learning of Approximate Manifolds (v0.23.0)

CLAM is a Rust/Python library for learning approximate manifolds from data.
It is designed to be fast, memory-efficient, easy to use, and scalable for big data applications.

CLAM provides utilities for fast search (CAKES) and anomaly detection (CHAODA).

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Installation

```shell
> python3 -m pip install "abd_clam==0.23.0"
```

## Usage

```python
from abd_clam.search import CAKES
from abd_clam.utils import synthetic_data

# Get the data.
data, _ = synthetic_data.bullseye()
# data is a numpy.ndarray in this case but it could just as easily be a
# numpy.memmap if your data do fit in RAM. We used numpy memmaps for the
# research, though they impose file-IO costs.

model = CAKES(data, 'euclidean')
# The CAKES class provides the functionality described in our
# [CHESS paper](https://arxiv.org/abs/1908.08551).

model.build(max_depth=50)
# Build the search tree to depth of 50.
# This method can be called again with a higher depth, if needed.

query, radius, k = data[0], 0.5, 10

rnn_results = model.rnn_search(query, radius)
# This is how we perform ranged nearest neighbors search with radius 0.5 around
# the query.

knn_results = model.knn_search(query, k)
# This is how we perform k-nearest neighbors search for the 10 nearest neighbors
# of the query.

# The results are returned as a dictionary whose keys are indices into the data
# array and whose values are the distance to the query.
```

<!-- TODO: Provide snippets for using CHAODA -->

## License

[MIT](LICENSE)

## Citation

TODO
