# CLAM: Clustering, Learning and Approximation with Manifolds

The Rust implementation of CLAM.

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Benchmarks

### CAKES: IEEE Big Data 2023

Take a look at the following scripts:

- `knn-benchmarks.sh`
- `scaling-benchmarks.sh`

## Components

This repository is a workspace that contains the following crates:

- `abd-clam`: The main CLAM library. See [here](abd-clam/README.md) for more information.
- `py-clam`: Python bindings for the `abd-clam` crate. See [here](py-clam/README.md) for more information.
- `distances`: A crate that provides various distance metrics. See [here](distances/README.md) for more information.

## License

- MIT

## Publications

- [CHESS](https://arxiv.org/abs/1908.08551)
- [CHAODA](https://arxiv.org/abs/2103.11774)

## Development

### Prerequisites

1. [`docker`](https://docs.docker.com/engine/install/) **or** [`podman`](https://podman.io/getting-started/installation)
2. [`hermit`](https://cashapp.github.io/hermit/usage/get-started/)

### Getting Started

1. Clone the repo.
2. CD into the folder.
3. Check `hermit` is installed correctly
   1. `hermit status` should show earthly and rust as being installed.
   2. `earthly --version` should return successfully.
   3. `cargo --version` should return successfully.
4. 

### Building

To work in this repo, the only tool you must have installed is [`hermit`](https://cashapp.github.io/hermit/usage/get-started/).
If you have `hermit` installed, once you clone this repo, you should have all of the tooling you need present automatically.

If you do not wish to use `hermit`

## Citation

TODO
