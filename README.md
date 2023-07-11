# CLAM: Clustered Learning of Approximate Manifolds (v0.18.0)

CLAM is a Rust/Python library for learning approximate manifolds from data.
It is designed to be fast, memory-efficient, easy to use, and scalable for big data applications.

CLAM provides utilities for fast search (CAKES) and anomaly detection (CHAODA).

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Usage

CLAM is a library crate so you can add it to your crate using `cargo add abd_clam@0.18.0`.

Here is a simple example of how to use CLAM to perform nearest neighbors search:

```rust
use symagen::random_data;

use abd_clam::{
    KnnAlgorithm, RnnAlgorithm, CAKES,
    PartitionCriteria,
    VecDataset,
};

/// Euclidean distance function.
///
/// This function is used to compute the distance between two points for the purposes
/// of this demo. You can use your own distance function instead. The required
/// signature is `fn(T, T) -> U` where `T` is the type of the points (must
/// implement `Send`, `Sync` and `Copy`) and `U` is a `Number` type (e.g. `f32`)
/// from the `distances` crate.
fn euclidean(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| a - b)
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt()
}

// Some parameters for generating random data.
let seed = 42;
let (cardinality, dimensionality) = (1_000, 10);
let (min_val, max_val) = (-1., 1.);

/// Generate some random data. You can use your own data here.
let data = random_data::random_f32(cardinality, dimensionality, min_val, max_val, seed);

// We will use the first point in data as our query, and we will perform
// RNN search with a radius of 0.05 and KNN search for the 10 nearest neighbors.
let (query, radius, k) = (data[0].clone(), 0.05, 10);

// We need the contents of data to be &[f32] instead of Vec<f32>. We will rectify this
// in CLAM by extending the trait bounds of some types in CLAM.
let data = data.iter().map(|v| v.as_slice()).collect::<Vec<_>>();

let name = "demo".to_string();
let data = VecDataset::new(name, data, euclidean, false);

// We will use the default partition criteria for this example. This will partition
// the data until each Cluster contains a single unique point.
let criteria = PartitionCriteria::default();

// The CAKES struct provides the functionality described in the CHESS paper.
let model = CAKES::new(data, Some(seed), criteria);

// We can now perform RNN search on the model.
let rnn_results: Vec<(usize, f32)> = model.rnn_search(&query, radius, RnnAlgorithm::Clustered);
assert!(!rnn_results.is_empty());

// We can also perform KNN search on the model.
let knn_results: Vec<(usize, f32)> = model.knn_search(&query, 10, KnnAlgorithm::RepeatedRnn);
assert!(knn_results.len() >= k);

// Both results are a Vec of 2-tuples where the first element is the index of the point
// in the dataset and the second element is the distance from the query point.
```

<!-- TODO: Provide snippets for using CHAODA -->

## License

- MIT

## References

- [CHESS](https://arxiv.org/abs/1908.08551)
- [CHAODA](https://arxiv.org/abs/2103.11774)

## Citation

TODO
