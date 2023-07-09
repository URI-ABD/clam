# CLAM: Clustered Learning of Approximate Manifolds (v0.15.0)

CLAM is a Rust/Python library for learning approximate manifolds from data.
It is designed to be fast, memory-efficient, easy to use, and scalable for big data applications.

CLAM provides utilities for fast search (CAKES) and anomaly detection (CHAODA).

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Usage

CLAM is a library crate so you can add it to your crate using:

```shell
> cargo add abd_clam@0.15.0
```

Here is a simple example of how to use CLAM to perform nearest neighbors search:

```rust
use abd_clam::cluster::PartitionCriteria;
use abd_clam::dataset::VecVec;
use abd_clam::cakes::CAKES;
use abd_clam::utils::synthetic_data;

fn euclidean(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn search() {
    // Get the data and queries.
    let seed = 42;
    let data: Vec<Vec<f32>> = synthetic_data::random_f32(100_000, 10, 0., 1., seed);
    let queries: Vec<Vec<f32>> = synthetic_data::random_f32(1_000, 10, 0., 1., 0);

    let dataset = VecVec::new(data, euclidean, "demo".to_string(), false);
    let criteria = PartitionCriteria::new(true).with_min_cardinality(1);
    let model = CAKES::new(dataset, Some(seed)).build(&criteria);
    // The CAKES struct provides the functionality described in our
    // [CHESS paper](https://arxiv.org/abs/1908.08551).

    let (query, radius, k) = (&queries[0], 0.05, 10);

    let rnn_results: Vec<(usize, f32)> = model.rnn_search(query, radius);
    // This is how we perform ranged nearest neighbors search with radius 0.05
    // around the query.

    let knn_results: Vec<(usize, f32)> = model.knn_search(query, 10);
    // This is how we perform k-nearest neighbors search for the 10 nearest
    // neighbors of query.

    // Both results are a Vec of 2-tuples where each tuple is the index and
    // distance to points in the data.

    todo!()
}
```

<!-- TODO: Provide snippets for using CHAODA -->

## License

[MIT](LICENSE)

## Citation

TODO
