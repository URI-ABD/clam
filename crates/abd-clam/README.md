# CLAM: Clustering, Learning and Approximation with Manifolds (v0.32.0)

The Rust implementation of CLAM.

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Usage

CLAM is a library crate so you can add it to your crate using `cargo add abd_clam@0.32.0`.

### Cakes: Nearest Neighbor Search

```rust
use abd_clam::{
    cakes::{cluster::Searchable, Algorithm},
    Ball, Cluster, FlatVec, Metric, Partition,
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
let rows: Vec<Vec<f32>> = symagen::random_data::random_tabular(
    cardinality,
    dimensionality,
    min_val,
    max_val,
    &mut rng,
);

// We will generate some random labels for each point.
let labels: Vec<bool> = rows.iter().map(|v| v[0] > 0.0).collect();

// We have to create a `Metric` object to encapsulate the distance function and its properties.
let metric = Metric::new(euclidean, false);

// We can create a `Dataset` object. We make it mutable here so we can reorder it after building the tree.
let data = FlatVec::new(rows, metric).unwrap();

// We can assign the labels as metadata to the dataset.
let data = data.with_metadata(&labels).unwrap();

// We define the criteria for building the tree to partition the `Cluster`s until each contains a single point.
let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;

// Now we create a tree.
let root = Ball::new_tree(&data, &criteria, Some(seed));

// We will use the origin as our query.
let query: Vec<f32> = vec![0.0; dimensionality];

// We can now perform Ranged Nearest Neighbors search on the tree.
let radius = 0.05;
let alg = Algorithm::RnnClustered(radius);
let rnn_results: Vec<(usize, f32)> = root.search(&data, &query, alg);

// KNN search is also supported.
let k = 10;

// The `KnnRepeatedRnn` algorithm starts RNN search with a small radius and increases it until it finds `k` neighbors.
let alg = Algorithm::KnnRepeatedRnn(k, 2.0);
let knn_results: Vec<(usize, f32)> = root.search(&data, &query, alg);

// The `KnnBreadthFirst` algorithm searches the tree in a breadth-first manner.
let alg = Algorithm::KnnBreadthFirst(k);
let knn_results: Vec<(usize, f32)> = root.search(&data, &query, alg);

// The `KnnDepthFirst` algorithm searches the tree in a depth-first manner.
let alg = Algorithm::KnnDepthFirst(k);
let knn_results: Vec<(usize, f32)> = root.search(&data, &query, alg);

// We can borrow the reordered labels from the model.
let labels: &[bool] = data.metadata();

// We can use the results to get the labels of the points that are within the
// radius of the query point.
let rnn_labels: Vec<bool> = rnn_results.iter().map(|&(i, _)| labels[i]).collect();

// We can use the results to get the labels of the points that are the k nearest
// neighbors of the query point.
let knn_labels: Vec<bool> = knn_results.iter().map(|&(i, _)| labels[i]).collect();
```

### Compression and Search

We also support compression of certain datasets and trees to reduce memory usage.
We can then perform compressed search on the compressed dataset without having to decompress the whole dataset.

```rust
use abd_clam::{
    adapter::ParBallAdapter,
    cakes::{cluster::ParSearchable, Algorithm, CodecData, SquishyBall},
    partition::ParPartition,
    Ball, Cluster, FlatVec, Metric, MetricSpace, Permutable,
};

// We will generate some random string data using the `symagen` crate.
let alphabet = "ACTGN".chars().collect::<Vec<_>>();
let seed_length = 100;
let seed_string = symagen::random_edits::generate_random_string(seed_length, &alphabet);
let penalties = distances::strings::Penalties::default();
let num_clumps = 10;
let clump_size = 10;
let clump_radius = 3_u32;
let inter_clump_distance_range = (clump_radius * 5, clump_radius * 7);
let len_delta = seed_length / 10;
let (metadata, data) = symagen::random_edits::generate_clumped_data(
    &seed_string,
    penalties,
    &alphabet,
    num_clumps,
    clump_size,
    clump_radius,
    inter_clump_distance_range,
    len_delta,
)
.into_iter()
.unzip::<_, _, Vec<_>, Vec<_>>();

// The dataset will use the `levenshtein` distance function from the `distances` crate.
let distance_fn = |a: &String, b: &String| distances::strings::levenshtein::<u16>(a, b);
let metric = Metric::new(distance_fn, true);
let data = FlatVec::new(data, metric.clone())
    .unwrap()
    .with_metadata(&metadata)
    .unwrap();

// We can serialize the dataset to disk without compression.
let temp_dir = tempdir::TempDir::new("readme-tests").unwrap();
let flat_path = temp_dir.path().join("strings.flat_vec");
let mut file = std::fs::File::create(&flat_path).unwrap();
bincode::serialize_into(&mut file, &data).unwrap();

// We build a tree from the dataset.
let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
let seed = Some(42);
let ball = Ball::par_new_tree(&data, &criteria, seed);

// We can serialize the tree to disk.
let ball_path = temp_dir.path().join("strings.ball");
let mut file = std::fs::File::create(&ball_path).unwrap();
bincode::serialize_into(&mut file, &ball).unwrap();

// We can adapt the tree and dataset to allow for compression and compressed search.
let (squishy_ball, codec_data) = SquishyBall::par_from_ball_tree(ball, data);

// The metadata types still need to be adjusted manually. We are working on a solution for this.
let squishy_ball = squishy_ball.with_metadata_type::<String>();
let codec_data = codec_data.with_metadata(&metadata).unwrap();

// We can serialize the compressed dataset to disk.
let codec_path = temp_dir.path().join("strings.codec_data");
let mut file = std::fs::File::create(&codec_path).unwrap();
bincode::serialize_into(&mut file, &codec_data).unwrap();

// We can serialize the compressed tree to disk.
let squishy_ball_path = temp_dir.path().join("strings.squishy_ball");
let mut file = std::fs::File::create(&squishy_ball_path).unwrap();
bincode::serialize_into(&mut file, &squishy_ball).unwrap();

// We can perform compressed search on the compressed dataset.
let query = &seed_string;
let radius = 2;
let alg = Algorithm::RnnClustered(radius);
let results: Vec<(usize, u16)> = squishy_ball.par_search(&codec_data, query, alg);
assert!(!results.is_empty());

let k = 10;
let alg = Algorithm::KnnRepeatedRnn(k, 2);
let results: Vec<(usize, u16)> = squishy_ball.par_search(&codec_data, query, alg);
assert_eq!(results.len(), k);

let alg = Algorithm::KnnBreadthFirst(k);
let results: Vec<(usize, u16)> = squishy_ball.par_search(&codec_data, query, alg);
assert_eq!(results.len(), k);

let alg = Algorithm::KnnDepthFirst(k);
let results: Vec<(usize, u16)> = squishy_ball.par_search(&codec_data, query, alg);
assert_eq!(results.len(), k);

// The dataset can be deserialized from disk.
let mut flat_data: FlatVec<String, u16, String> =
    bincode::deserialize_from(std::fs::File::open(&flat_path).unwrap()).unwrap();
// Since functions cannot be serialized, we have to set the metric manually.
flat_data.set_metric(metric.clone());

// The tree can be deserialized from disk.
let ball: Ball<String, u16, FlatVec<String, u16, String>> =
    bincode::deserialize_from(std::fs::File::open(&ball_path).unwrap()).unwrap();

// The compressed dataset can be deserialized from disk.
let mut codec_data: CodecData<String, u16, String> =
    bincode::deserialize_from(std::fs::File::open(&codec_path).unwrap()).unwrap();
// The metric has to be set manually.
codec_data.set_metric(metric.clone());

// The compressed tree can be deserialized from disk.
// You will forgive the long type signature.
let squishy_ball: SquishyBall<
    String,
    u16,
    FlatVec<String, u16, String>,
    CodecData<String, u16, String>,
    Ball<String, u16, FlatVec<String, u16, String>>,
> = bincode::deserialize_from(
    std::fs::File::open(&squishy_ball_path)
        .map_err(|e| e.to_string())
        .unwrap(),
)
.unwrap();
```

### Chaoda: Anomaly Detection

TODO ...

## License

- MIT
