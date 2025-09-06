# CLAM: Clustering, Learning and Approximation with Manifolds

The Rust implementation of CLAM.

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Rust Crates and Python Packages

This repository is a workspace that contains the following crates:

- [`abd-clam`](crates/abd-clam/README.md): The main CLAM library.
- [`distances`](crates/distances/README.md): Provides various distance functions and the `Number` trait.
- [`shell`](crates/shell/README.md): The CLI for interacting with the CLAM library.

and the following Python packages:

- [`abd-distances`](python/distances/README.md): A Python wrapper for the `distances` crate, providing drop-in replacements for `scipy.spatial.distance`.

## Publications

- [CHESS](https://arxiv.org/abs/1908.08551): Ranged Nearest Neighbors Search
- [CHAODA](https://arxiv.org/abs/2103.11774): Anomaly Detection
- [CAKES](https://arxiv.org/abs/2309.05491): K-NN Search
- [PANCAKES](https://arxiv.org/abs/2409.12161): Compression and Compressive Search

## Citation

TODO
