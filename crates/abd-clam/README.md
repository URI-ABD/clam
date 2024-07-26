# CLAM: Clustering, Learning and Approximation with Manifolds (v0.31.0)

The Rust implementation of CLAM.

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Usage

CLAM is a library crate so you can add it to your crate using `cargo add abd_clam@0.31.0`.

### Cakes: Nearest Neighbor Search

```rust
use abd_clam::{
    Ball, Cluster, FlatVec, Metric, Partition,
    cakes::{Algorithm, Searchable},
};
use rand::prelude::*;

/// The distance function with with to perform clustering and search.
///
/// We use the `distances` crate for the distance function.
fn euclidean(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    distances::simd::euclidean_f32(x, y)
}

/// Generate some random data. You can use your own data here.
///
/// CLAM can handle arbitrarily large datasets. We use a small one here for
/// demonstration.
///
/// We use the `symagen` crate for generating interesting datasets for examples
/// and tests.
let seed = 42;
let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
let (cardinality, dimensionality) = (1_000, 10);
let (min_val, max_val) = (-1.0, 1.0);
let data: Vec<Vec<f32>> = symagen::random_data::random_tabular(
    cardinality,
    dimensionality,
    min_val,
    max_val,
    &mut rng,
);

// We will generate some random labels for each point.
let labels: Vec<bool> = data.iter().map(|v| v[0] > 0.0).collect();

// We have to create a `Metric` object to encapsulate the distance function and its properties.
let metric = Metric::new(euclidean, false);

// We can create a `Dataset` object. We make it mutable here so we can reorder it after building the tree.
let dataset = FlatVec::new(data, metric).unwrap();

// We can assign the labels as metadata to the dataset.
let dataset = dataset.with_metadata(labels).unwrap();

// We make the dataset mutable so we can reorder it after building the tree.
let mut dataset = dataset;

// We define the criteria for building the tree to partition the `Cluster`s until each contains a single point.
let criteria = |c: &Ball<f32>| c.cardinality() > 1;

// Now we create a tree and reorder the dataset.
let (root, _) = Ball::new_tree_and_permute(&mut dataset, &criteria, Some(seed));

// We will use the origin as our query.
let query: Vec<f32> = vec![0.0; dimensionality];

// We can now perform RNN search on the tree.
let alg = Algorithm::RnnClustered(0.05);
let rnn_results: Vec<(usize, f32)> = root.search(&dataset, &query, alg);

// KNN search is also supported.
let k = 10;

// The `KnnRepeatedRnn` algorithm starts RNN search with a small radius and increases it until it finds `k` neighbors.
let alg = Algorithm::KnnRepeatedRnn(k, 2.0);
let knn_results: Vec<(usize, f32)> = root.search(&dataset, &query, alg);

// The `KnnBreadthFirst` algorithm searches the tree in a breadth-first manner.
let alg = Algorithm::KnnBreadthFirst(k);
let knn_results: Vec<(usize, f32)> = root.search(&dataset, &query, alg);

// The `KnnDepthFirst` algorithm searches the tree in a depth-first manner.
let alg = Algorithm::KnnDepthFirst(k);
let knn_results: Vec<(usize, f32)> = root.search(&dataset, &query, alg);

// We can borrow the reordered labels from the model.
let labels: &[bool] = dataset.metadata();

// We can use the results to get the labels of the points that are within the
// radius of the query point.
let rnn_labels: Vec<bool> = rnn_results.iter().map(|&(i, _)| labels[i]).collect();

// We can use the results to get the labels of the points that are the k nearest
// neighbors of the query point.
let knn_labels: Vec<bool> = knn_results.iter().map(|&(i, _)| labels[i]).collect();
```

### Chaoda: Anomaly Detection

TODO ...

## License

- MIT
