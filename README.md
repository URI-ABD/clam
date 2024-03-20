# CLAM: Clustering, Learning and Approximation with Manifolds

The Rust implementation of CLAM.

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Components

This repository is a workspace that contains the following crates:

- `abd-clam`: The main CLAM library. See [here](crates/abd-clam/README.md) for more information.
- `distances`: Provides various distance functions and the `Number` trait. See [here](crates/distances/README.md) for more information.

and the following Python packages:

- `abd-distances`: A Python wrapper for the `distances` crate, providing drop-in replacements for distance function `scipy.spatial.distance`. See [here](python/distances/README.md) for more information.

## License

- MIT

## Publications

- [CHESS](https://arxiv.org/abs/1908.08551)
- [CHAODA](https://arxiv.org/abs/2103.11774)

## Citation

TODO
