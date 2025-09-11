# CLAM: Clustering, Learning and Approximation with Manifolds

The Rust implementation of CLAM.

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Rust Crates and Python Packages

This repository is a workspace that contains the following crates:

- `abd-clam`: The main CLAM library. See [here](crates/abd-clam/README.md) for more information.
- `distances`: Provides various distance functions and the `Number` trait. See [here](crates/distances/README.md) for more information.
- `clam-mbed`: A CLI for running the dimensionality reduction algorithms. See [here](tools/mbed/README.md) for more information.
- `shell`: The CLI for interacting with the CLAM library. See [here](crates/shell/README.md) for more information.

and the following Python packages:

- `abd-distances`: A Python wrapper for the `distances` crate, providing drop-in replacements for distance function `scipy.spatial.distance`. See [here](python/distances/README.md) for more information.

## Reproducing Results from Papers

This repository contains CLI tools to reproduce results from some of our papers.

### CAKES

This paper is currently under review at SIMODS.
See [here](benches/cakes/README.md) for running Rust code to reproduce the results for the CAKES algorithms, and [here](benches/py-cakes/README.md) for running some Python code to generate plots from the results of running the Rust code.

### CHAODA

TODO ...

### PANCAKES

This paper was published at IEEE BigData 2024.
See [here](benches/pancakes/README.md) for running Rust code to reproduce the results for the PANCAKES algorithms.

### MUSALS

This paper is a work in progress.
See [here](benches/musals/README.md) for running Rust code to reproduce the results for the MUSALS algorithms.

## Publications

- [CHESS](https://arxiv.org/abs/1908.08551): Ranged Nearest Neighbors Search
- [CHAODA](https://arxiv.org/abs/2103.11774): Anomaly Detection
- [CAKES](https://arxiv.org/abs/2309.05491): K-NN Search
- [PANCAKES](https://arxiv.org/abs/2409.12161): Compression and Compressive Search

## Citation

TODO
