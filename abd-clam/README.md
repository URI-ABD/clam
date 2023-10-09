# CLAM: Clustering, Learning and Approximation with Manifolds (v0.22.3)

The Rust implementation of CLAM.

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Usage

CLAM is a library crate so you can add it to your crate using `cargo add abd_clam@0.22.3`.

### Cakes: Nearest Neighbor Search

```rust
use symagen::random_data;

use abd_clam::{knn, rnn, Cakes, PartitionCriteria, VecDataset};

/// Euclidean distance function.
///
/// This function is used to compute the distance between two points for the purposes
/// of this demo. You can use your own distance function instead. The required
/// signature is `fn(&I, &I) -> U` where `I` is the type of the points (must
/// implement the `Instance` trait) and `U` is a `Number` type (e.g. `f32`)
/// from the `distances` crate.
fn euclidean(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| a - b)
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt()
}

// Some parameters for generating random data.
let seed = 42;
let (cardinality, dimensionality) = (10_000, 10);
let (min_val, max_val) = (-1.0, 1.0);

/// Generate some random data. You can use your own data here.
let data: Vec<Vec<f32>> = random_data::random_f32(
    cardinality,
    dimensionality,
    min_val,
    max_val,
    seed,
);

// We will use the first point in data as our query, and we will perform
// RNN search with a radius of 0.05 and KNN search for the 10 nearest neighbors.
let query: Vec<f32> = data[0].clone();
let radius: f32 = 0.05;
let k = 10;

// The name of the dataset.
let name = "demo".to_string();

// We will assume that our distance function is cheap to compute.
let is_metric_expensive = false;

// The metric function itself will be given to Cakes.
let data = VecDataset::<Vec<f32>, f32>::new(
    name,
    data,
    euclidean,
    is_metric_expensive,
);

// We will use the default partition criteria for this example. This will partition
// the data until each Cluster contains a single unique point.
let criteria = PartitionCriteria::default();

// The Cakes struct provides the functionality described in the CHESS paper.
// This line performs a non-trivial amount of work.
let model = Cakes::new(data, Some(seed), &criteria);

// Note that the dataset has been reordered to improve search performance.

// We will soon add the ability to save and load models, but for now we will
// just use the model we just created.

// We can now perform RNN search on the model.
let rnn_results: Vec<(usize, f32)> = model.rnn_search(
    &query,
    radius,
    rnn::Algorithm::Clustered,
);
assert!(!rnn_results.is_empty());

// We can also perform KNN search on the model.
let knn_results: Vec<(usize, f32)> = model.knn_search(
    &query,
    k,
    knn::Algorithm::default(),
);
assert!(knn_results.len() >= k);

// Both results are a Vec of 2-tuples where the first element is the index of
// the point in the dataset and the second element is the distance from the
// query point.
```

### Chaoda: Anomaly Detection

TODO ...

## License

- MIT
