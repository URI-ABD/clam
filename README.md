# URI-ABD: Clustered Learning of Approximate Manifolds

## Installation

### Docker

```
docker build -t clam .
```

### Python

```bash
python3 -m pip install pyclam
```

## Usage

### Docker

```
docker run clam --help
```

### Python Scripting

```python
from pyclam import Manifold
from pyclam import CAKES
from pyclam import criterion
from pyclam.utils import synthetic_datasets

# Get the data.
data, _ = synthetic_datasets.bullseye()
# data is a numpy.ndarray in this case but it could just as easily be a numpy.memmap if your data does fit in RAM.
# We used numpy memmaps for the research, though they impose file-IO costs.

search = CAKES(data, 'euclidean')
# The Search class provides the functionality described in our CHESS paper.
# TODO: Provide link to CHESS paper

search.build(max_depth=10)
# Build the search tree to depth of 10.
# This method can be called again with a higher depth, if needed.

query, radius = data[0], 0.5
rnn_results = search.rnn(query, radius)
# This is how we perform rho-nearest neighbors search with radius 0.5 around the query.

knn_results = search.knn(query, 10)
# This is how to perform k-nearest neighbors search for the 10 nearest neighbors of query.

# TODO: Provide snippets for using CHAODA

# You can also directly use the Manifold functionality provided by CLAM.

manifold = Manifold(data, 'euclidean')
# Any metric allowed by scipy's cdist function is allowed in Manifold.
# You can also define your own distance function. It will work so long as scipy allows it.

manifold.build(
    criterion.MaxDepth(20),  # build the tree to a maximum depth of 20
    criterion.MinRadius(0.25),  # clusters with radius less than 0.25 cannot be partitioned.
    criterion.Layer(6),  # use the clusters ad depth 6 to build a Graph.
    criterion.Leaves(),  # use the leaves of the tree to build another Graph.
)
# Manifold.build can optionally take any number of criteria.
# pyclam.criterion defines some criteria that we have used in research.
# You are free to define your own.
# Take a look at pyclam/criterion.py for hints of how to define custom criteria.
```

The Manifold class relies on the Graph and Cluster classes.
You can import these and work with them directly if you so choose.
The classes and methods are all very well documented.
Go crazy.

## Contributing

Pull requests and bug reports are welcome.
For major changes, please open an issue to discuss what you would like to change.

## License

[MIT](LICENSE)
