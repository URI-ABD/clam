# CLAM: Clustering, Learning and Approximation with Manifolds (v0.32.0)

The Rust implementation of CLAM.

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Usage

CLAM is a library crate so you can add it to your crate using `cargo add abd_clam@0.32.0`.

## Features

This crate provides the following features:

- `serde`: Enables serialization and deserialization using `serde` and `databuf`.
- `musals`: Enables multiple sequence alignment using the `musals` module.
- `all`: Enables `serde` and `musals` features.
- `profile`: Enables profiling using the `profi` crate.

### `Cakes`: Nearest Neighbor Search

```rust
use abd_clam::{
    cakes::{self, SearchAlgorithm},
    Ball, Cluster, Partition,
};
use rand::prelude::*;

// Generate some random data.
let seed = 42;
let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
let (cardinality, dimensionality) = (1_000, 10);
let (min_val, max_val) = (-1.0, 1.0);
let data: Vec<Vec<f32>> =
    symagen::random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rng);

// We use the `Euclidean` metric for this example.
fn metric(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    x.iter().zip(y.iter()).map(|(&a, &b)| a - b).map(|v| v * v).sum::<f32>().sqrt()
}

// We define the criteria for building the tree to partition the `Cluster`s
// until each contains a single point.
let criteria = |c: &Ball<_>| c.cardinality() > 1;

// Now we create a tree.
let root = Ball::new_tree(&data, &metric, &criteria);

// We will use the origin as our query.
let query = vec![0_f32; dimensionality];

// We can now perform Ranged Nearest Neighbors search on the tree.
let radius = 0.05;
let alg = cakes::RnnClustered(radius);
let rnn_results: Vec<(usize, f32)> = alg.search(&data, &metric, &root, &query);

// KNN search is also supported.
let k = 10;

// The `KnnRepeatedRnn` algorithm starts RNN search with a small radius and
// increases it until it finds `k` neighbors.
let alg = cakes::KnnRepeatedRnn(k, 2.0);
let knn_results: Vec<(usize, f32)> = alg.search(&data, &metric, &root, &query);

// The `KnnBreadthFirst` algorithm searches the tree in a breadth-first manner.
let alg = cakes::KnnBreadthFirst(k);
let knn_results: Vec<(usize, f32)> = alg.search(&data, &metric, &root, &query);

// The `KnnDepthFirst` algorithm searches the tree in a depth-first manner.
let alg = cakes::KnnDepthFirst(k);
let knn_results: Vec<(usize, f32)> = alg.search(&data, &metric, &root, &query);
```

### `PanCakes`: Compression and Compressive Search

We also support compression of certain datasets to reduce memory usage.
We can perform compressed search on the compressed dataset without having to decompress the whole dataset.

TODO: Add example.

<!-- ```rust
use abd_clam::{
    cakes::{self, ParSearchAlgorithm},
    cluster::{adapter::ParBallAdapter, ClusterIO, ParPartition},
    dataset::{AssociatesMetadataMut, DatasetIO},
    metric::Levenshtein,
    musals::{Aligner, CostMatrix, Sequence},
    pancakes::{CodecData, SquishyBall},
    Ball, Cluster, Dataset, FlatVec,
};

// We need an aligner to align the sequences for compression and decompression.

// We will be generating DNA/RNA sequence data for this example so we will use
// the default cost matrix for DNA sequences.
let cost_matrix = CostMatrix::<u16>::default();
let aligner = Aligner::new(&cost_matrix, b'-');

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
    .map(|(m, d)| (m, Sequence::new(d, Some(&aligner))))
    .unzip::<_, _, Vec<_>, Vec<_>>();

// We create a `FlatVec` dataset from the sequence data and assign metadata.
let data = FlatVec::new(data).unwrap().with_metadata(&metadata).unwrap();

// The dataset will use the `levenshtein` distance metric.
let metric = Levenshtein;

// We can serialize the dataset to disk without compression.
let temp_dir = tempdir::TempDir::new("readme-tests").unwrap();
let flat_path = temp_dir.path().join("strings.flat_vec");
data.write_to(&flat_path).unwrap();

// We build a tree from the dataset.
let criteria = |c: &Ball<_>| c.cardinality() > 1;
let seed = Some(42);
let ball = Ball::par_new_tree(&data, &metric, &criteria, seed);

// We can serialize the tree to disk.
let ball_path = temp_dir.path().join("strings.ball");
ball.write_to(&ball_path).unwrap();

// We can adapt the tree and dataset to allow for compression and compressed search.
let (squishy_ball, codec_data) = SquishyBall::par_from_ball_tree(ball, data, &metric);

// The metadata type still need to be adjusted manually. We are working on a solution for this.
let codec_data = codec_data.with_metadata(&metadata).unwrap();

// We can serialize the compressed dataset to disk.
let codec_path = temp_dir.path().join("strings.codec_data");
codec_data.write_to(&codec_path).unwrap();
// Note that serialization of `Sequence` types does not store the `Aligner`.

// We can serialize the compressed tree to disk.
let squishy_ball_path = temp_dir.path().join("strings.squishy_ball");
squishy_ball.write_to(&squishy_ball_path).unwrap();

// We can perform compressive search on the compressed dataset.
let query = &Sequence::new(seed_string, Some(&aligner));
let radius = 2;
let k = 10;

let alg = cakes::RnnClustered(radius);
let results: Vec<(usize, u16)> = alg.par_search(&codec_data, &metric, &squishy_ball, query);
assert!(!results.is_empty());

let alg = cakes::KnnRepeatedRnn(k, 2);
let results: Vec<(usize, u16)> = alg.par_search(&codec_data, &metric, &squishy_ball, query);
assert_eq!(results.len(), k);

let alg = cakes::KnnBreadthFirst(k);
let results: Vec<(usize, u16)> = alg.par_search(&codec_data, &metric, &squishy_ball, query);
assert_eq!(results.len(), k);

let alg = cakes::KnnDepthFirst(k);
let results: Vec<(usize, u16)> = alg.par_search(&codec_data, &metric, &squishy_ball, query);
assert_eq!(results.len(), k);

// The dataset can be deserialized from disk.
let flat_data: FlatVec<Sequence<u16>, String> = FlatVec::read_from(&flat_path).unwrap();

// The tree can be deserialized from disk.
let ball: Ball<u16> = Ball::read_from(&ball_path).unwrap();

// The compressed dataset can be deserialized from disk.
let codec_data: CodecData<Sequence<u16>, String> = CodecData::read_from(&codec_path).unwrap();
// Since the serialization of `Sequence` types does not store the `Aligner`, we
// need to manually set it.
let codec_data = codec_data.transform_centers(|s| s.with_aligner(&aligner));

// The compressed tree can be deserialized from disk.
let squishy_ball: SquishyBall<u16, Ball<_>> = SquishyBall::read_from(&squishy_ball_path).unwrap();
``` -->

### `Musals`: Multiple Sequence Alignment at Scale

For sequence data, we support multiple sequence alignment (MSA) creation and evaluation using the `musals` module.

TODO: Add example.

### `Chaoda`: Unsupervised Anomaly Detection

The `chaoda` module provides unsupervised anomaly detection algorithms based on the CLAM framework.

TODO: Add example.

## License

- MIT
