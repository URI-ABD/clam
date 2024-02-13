# CLAM: Clustering, Learning and Approximation with Manifolds (v0.28.0)

The Rust implementation of CLAM.

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Usage

CLAM is a library crate so you can add it to your crate using `cargo add abd_clam@0.28.0`.

### Cakes: Nearest Neighbor Search

```rust
use abd_clam::{knn, rnn, Cakes, PartitionCriteria, VecDataset, UniBall};
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

// We will use the origin as our query.
let query: Vec<f32> = vec![0.0; dimensionality];

// RNN search will use a radius of 0.05.
let radius: f32 = 0.05;

// KNN search will find the 10 nearest neighbors.
let k = 10;

// The name of the dataset.
let name = "demo".to_string();

// We will assume that our distance function is cheap to compute.
let is_metric_expensive = false;

// We create the dataset from the data and distance function.
let dataset = VecDataset::new(
    name,
    data,
    euclidean,
    is_metric_expensive,
);

// At this point, `dataset` has taken ownership of the `data`.

// The default metadata is the indices of the points in the dataset. We will,
// however, use our random labels as metadata.
let dataset = dataset
    .assign_metadata(labels)
    .unwrap_or_else(|_| unreachable!("We made sure that there are as make labels as points."));

// At this point, `dataset` has also taken ownership of `labels`.

// We will use the default partition criteria for this example. This will partition
// the data until each Cluster contains a single unique point.
let criteria = PartitionCriteria::<f32, UniBall<f32>>::default();

// The Cakes struct provides the functionality described in the paper.
// We use a single shard here because the demo data is small.
let model = Cakes::new(dataset, Some(seed), &criteria);
// This line performs a non-trivial amount of work. #understatement

// At this point, the dataset has been reordered to improve search performance.

// We can now perform RNN search on the model.
let rnn_results: Vec<(usize, f32)> = model.rnn_search(
    &query,
    radius,
    rnn::Algorithm::default(),
);

// We can also perform KNN search on the model.
let knn_results: Vec<(usize, f32)> = model.knn_search(
    &query,
    k,
    knn::Algorithm::default(),
);

// Both results are a Vec of 2-tuples where the first element is the index of
// the point in the dataset and the second element is the distance from the
// query point.

// We can borrow the reordered labels from the model.
let labels: &[bool] = model.shards()[0].metadata();

// We can use the results to get the labels of the points that are within the
// radius of the query point.
let rnn_labels: Vec<bool> = rnn_results.iter().map(|&(i, _)| labels[i]).collect();

// We can use the results to get the labels of the points that are the k nearest
// neighbors of the query point.
let knn_labels: Vec<bool> = knn_results.iter().map(|&(i, _)| labels[i]).collect();

// TODO: Add snippets for saving/loading models.
```

### Chaoda: Anomaly Detection

TODO ...

## License

- MIT
