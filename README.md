# CLAM: Clustered Learning of Approximate Manifolds (v0.11.0)

CLAM is a Rust/Python library for learning approximate manifolds from data.
It is designed to be fast, memory-efficient, easy to use, and scalable for big data applications.

CLAM provides utilities for fast search (CAKES) and anomaly detection (CHAODA).

## Installation

### Python

```shell
> python3 -m pip install abd_clam
```

### Rust

```shell
> cargo add abd_clam
```

## Usage

### Python

```python
from abd_clam.search import CAKES
from abd_clam.utils import synthetic_datasets

# Get the data.
data, _ = synthetic_datasets.bullseye()
# data is a numpy.ndarray in this case but it could just as easily be a numpy.memmap if your data do fit in RAM.
# We used numpy memmaps for the research, though they impose file-IO costs.

model = CAKES(data, 'euclidean')
# The Search class provides the functionality described in our [CHESS paper](https://arxiv.org/abs/1908.08551).

model.build(max_depth=50)
# Build the search tree to depth of 50.
# This method can be called again with a higher depth, if needed.

query, radius = data[0], 0.5

rnn_results = model.rnn_search(query, radius)
# This is how we perform ranged nearest neighbors search with radius 0.5 around the query.
# The results are returned as a dictionary whose keys are indices into the data array and whose values are the distance to the query.

knn_results = model.knn_search(query, 10)
# This is how we perform k-nearest neighbors search for the 10 nearest neighbors of query.

# TODO: Provide snippets for using CHAODA
```

## Contributing

Pull requests and bug reports are welcome.
For major changes, please open an issue to discuss what you would like to change.

## License

[MIT](LICENSE)

## Citation

TODO
