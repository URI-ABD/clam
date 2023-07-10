# CLAM: Clustered Learning of Approximate Manifolds (v0.16.1)

CLAM is a Rust/Python library for learning approximate manifolds from data.
It is designed to be fast, memory-efficient, easy to use, and scalable for big data applications.

CLAM provides utilities for fast search (CAKES) and anomaly detection (CHAODA).

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Usage

CLAM is a library crate so you can add it to your crate using `cargo add abd_clam@0.16.1`.

Here is a simple example of how to use CLAM to perform nearest neighbors search:

```rust
use symagen::random_data;

use abd_clam::{
    cakes::{KnnAlgorithm, RnnAlgorithm, CAKES},
    cluster::PartitionCriteria,
    dataset::VecVec,
};

fn euclidean(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

// Get the data and queries. We will generate some random data for this demo.
let seed = 42;
let (cardinality, dimensionality) = (1_000, 10);
let (min_val, max_val) = (-1., 1.);

let data = random_data::random_f32(cardinality, dimensionality, min_val, max_val, seed);
let data = data.iter().map(|v| v.as_slice()).collect::<Vec<_>>();

let dataset = VecVec::new(data.clone(), euclidean, "demo".to_string(), false);
let criteria = PartitionCriteria::new(true).with_min_cardinality(1);
let model = CAKES::new(dataset, Some(seed), criteria);

// The CAKES struct provides the functionality described in the CHESS paper.

let (query, radius, k) = (&data[0], 0.05, 10);

let rnn_results: Vec<(usize, f32)> = model.rnn_search(query, radius, RnnAlgorithm::Clustered);
assert!(!rnn_results.is_empty());
// This is how we perform ranged nearest neighbors search with radius 0.05
// around the query.

let knn_results: Vec<(usize, f32)> = model.knn_search(query, 10, KnnAlgorithm::RepeatedRnn);
assert!(knn_results.len() >= k);
// This is how we perform k-nearest neighbors search for the 10 nearest
// neighbors of query.

// Both results are a Vec of 2-tuples where each tuple is the index and
// distance to points in the data.
```

<!-- TODO: Provide snippets for using CHAODA -->

## License

- MIT

## References

- [CHESS](https://arxiv.org/abs/1908.08551)
- [CHAODA](https://arxiv.org/abs/2103.11774)

## Citation

TODO
