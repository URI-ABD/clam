# CLAM: Clustering, Learning and Approximation with Manifolds (v0.34.0)

The Rust implementation of CLAM.

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Usage

CLAM is a library crate so you can add it to your crate using `cargo add abd_clam@0.34.0`.

## Features

This crate provides the following features:

- `serde`: Enables serialization and deserialization using `serde` and `databuf`.
- `musals`: Enables multiple sequence alignment using the `musals` module.
- `pancakes`: Enables compression and compressive search using the `pancakes` module.
- `all`: Enables `serde`, `musals`, and `pancakes` features.
- `profile`: Enables profiling using the `profi` crate.

### `Cakes`: Nearest Neighbor Search

```rust
use abd_clam::{
    DistanceValue, NamedAlgorithm, Tree,
    cakes::{self, Search},
};
use rand::prelude::*;

// Generate some random data.
let seed = 42;
let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
let (cardinality, dimensionality) = (1_000, 10);
let (min_val, max_val) = (-1.0, 1.0);
let data: Vec<Vec<f32>> = {
    let distr = rand::distr::Uniform::new_inclusive(min_val, max_val).unwrap();
    (0..cardinality).map(|_| (0..dimensionality).map(|_| rng.sample(distr)).collect()).collect()
};

// We use the `Euclidean` metric for this example.
fn metric(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    x.iter().zip(y.iter()).map(|(&a, &b)| a - b).map(|v| v * v).sum::<f32>().sqrt()
}

// We will create a tree from the dataset using the default partition strategy.
let tree = Tree::new_minimal(data.clone(), metric).unwrap_or_else(|err| unreachable!("Data was non-empty. Error: {err}"));
// The tree can also be built in parallel using `par_new_minimal` instead of `new_minimal`.
let tree = Tree::par_new_minimal(data, metric).unwrap_or_else(|err| unreachable!("Data was non-empty. Error: {err}"));

// We will use the origin as our query for this example.
let query = vec![0_f32; dimensionality];

// We support a variety of algorithms for nearest neighbor search on the tree.
let algorithms = vec![
    cakes::Cakes::from(cakes::RnnLinear::new(0.1)), // Ranged search with radius 0.1 using a linear scan.
    cakes::Cakes::from(cakes::RnnChess::new(0.1)), // Ranged search with radius 0.1 using the CHESS algorithm.
    cakes::Cakes::from(cakes::KnnLinear::new(10)), // KNN search for 10 neighbors using a linear scan.
    cakes::Cakes::from(cakes::KnnRrnn::new(10)), // KNN search for 10 neighbors using the Repeated RNN algorithm.
    cakes::Cakes::from(cakes::KnnBfs::new(10)), // KNN search for 10 neighbors using the Breadth-First Sieve algorithm.
    cakes::Cakes::from(cakes::KnnDfs::new(10)), // KNN search for 10 neighbors using the Depth-First Sieve algorithm.
];

for alg in &algorithms {
    // The `search` method returns a vector of `(index, distance)` pairs for the neighbors found.
    // The indices from `results` can be used to retrieve the neighbors from the tree via `tree.items()[index]`.
    let results: Vec<(usize, f32)> = alg.search(&tree, &query);

    // The algorithms also have parallel versions that can be used with `par_search` instead of `search`.
    let par_results = alg.par_search(&tree, &query);
}
```

### `PanCakes`: Compression and Compressive Search

We also support compression of certain datasets to reduce memory usage.
We can perform compressed search on the compressed dataset without having to decompress the whole dataset.

TODO: Add example.

### `Musals`: Multiple Sequence Alignment at Scale

For sequence data, we support multiple sequence alignment (MSA) creation and evaluation using the `musals` module.

TODO: Add example.

### `Chaoda`: Unsupervised Anomaly Detection

The `chaoda` module provides unsupervised anomaly detection algorithms based on the CLAM framework.

TODO: Add example.

## The `Clam-Shell` CLI Tool

The `clam-shell` binary crate provides a command-line interface for performing various operations using the CLAM library, such as building trees, performing nearest neighbor search, and more.

Install the `clam-shell` binary:

```bash
cargo install --path ./crates/abd-clam --features=shell --bin clam-shell
```

Run the following command to see the available subcommands and options:

```bash
clam-shell --help
```

There are a number of parameters that can be set globally for all subcommands, as well as parameters specific to each subcommand. The global parameters include:

- `-i, --inp-path <INP_PATH>`: The path to the input file or directory, depending on the subcommand.
- `-o, --out-dir <OUT_PATH>`: The path to the output directory where results will be saved.
- `-s, --seed <SEED>`: An optional random seed for reproducibility.
- `-l, --log-name <LOG_NAME>`: An optional name for the log file. The default is `shell.log`.

The following subcommands are available:

- `build`: Build a tree from a dataset.
- `cakes`: Perform and benchmark nearest neighbor search using the `cakes` algorithms.
- `pancakes`: Perform and benchmark compression and compressive search using the `pancakes` module.
- `musals`: Perform multiple sequence alignment and evaluation using the `musals` module.
- `generate-data`: Generate synthetic datasets for testing and benchmarking.

### Build

Run the following command to see the options for the `build` subcommand:

```bash
clam-shell build --help
```

This requires the `--inp-path` parameter to specify the path to the input dataset file, and the `--out-dir` parameter to specify the output directory where the built tree will be saved. This command has the following additional parameters:

- `-t, --data-type <DATA_TYPE>`: The type of the input dataset. Supported values are `string` and `aligned` for sequence data read from FASTA files, and `f64` and `f32` for vector data read from NPY files.
- `-n, --num-samples <NUM_SAMPLES>`: An optional parameter to specify the number of samples to use from the input dataset to build the tree. If not specified, the entire dataset will be used.
- `-m, --metric <METRIC>`: The distance metric to use for building the tree. Supported values are `levenshtein` for string data, and `euclidean` and `cosine` for vector data.
- `-p, --partition-strategy <PARTITION_STRATEGY>`: The partition strategy to use for building the tree. See the documentation for the `PartitionStrategy` enum in `abd_clam` for the supported values and their descriptions. If not specified, the default partition strategy will be used.

Here is an example command to build a tree from a FASTA file containing protein sequences using the Levenshtein distance metric and the default partition strategy:

```bash
clam-shell \
    --inp-path ../data/string-data/pfam-a/pfam_10000.fasta \
    --out-dir ../data/string-data/pfam-a/msa-results/10k \
    --seed 42 \
    --log-name build_pfam_10k \
    build \
    --data-type string \
    --metric levenshtein
```

### Cakes

Run the following command to see the options for the `cakes` subcommand:

```bash
clam-shell cakes --help
```

This requires the `--inp-path` parameter to specify the path to the input tree file, and the `--out-dir` parameter to specify the output directory where the search results will be saved. This command has the following additional parameters:

- `-f, --out-fmt <OUT_FMT>`: The output format for the search results. Supported values are `json` and `yaml`. The default is `json`.
- `-q, --queries-path <QUERIES_PATH>`: The path to the file containing the query points. The format of the query file should be the same as the input dataset file used to build the tree.
- `-n, --num-queries <NUM_QUERIES>`: An optional parameter to specify the number of queries to run. If not specified, all queries in the query file will be used.
- `-a, --algorithm <ALGORITHM>`: The search algorithm to use. All variants of the `Cakes` enum in the `cakes` module are supported. For example, `knn-dfs::k=10` would specify the `KnnDfs` algorithm with `k=10` neighbors.

This command has the following subcommands for different use cases:

- `search`: Run an algorithm for nearest neighbor search on the input tree.
- `bench`: Benchmark the performance of a search algorithm on the input tree.
- `frontier`: Compute a throughput vs quality frontier for a search algorithm on the input tree.

#### Search

Run the following command to see the options for the `search` subcommand:

```bash
clam-shell cakes search --help
```

The `search` subcommand has no additional parameters.

Here is an example command to run a KNN search for 10 neighbors using the Depth-First Sieve algorithm on the previously built tree from the Pfam dataset, using 10 queries from the same dataset:

```bash
clam-shell \
    --inp-path "../data/string-data/pfam-a/msa-results/10k/trees/tree-pfam_10000-string-levenshtein-span-reduction-factor::sqrt2.tree.bin" \
    --out-dir ../data/string-data/pfam-a/msa-results/10k/search-results \
    --seed 42 \
    --log-name cakes_search_pfam_10k \
    cakes \
    --out-fmt yaml \
    --queries-path ../data/string-data/pfam-a/pfam_10000.fasta \
    --num-queries 10 \
    --algorithm "knn-dfs::k=10" \
    search
```

#### Bench

Run the following command to see the options for the `bench` subcommand:

```bash
clam-shell cakes bench --help
```

The `bench` subcommand has the following additional parameters:

- `-c, --quality-metrics <QUALITY_METRICS>`: Zero or more quality metrics to compute for the search results. Supported values are `recall` and `relative-distance-error`. If not specified, both will be computed.

This command will always measure the throughput of the search algorithm, and will compute the specified quality metrics by comparing the search results to the ground truth nearest neighbors obtained via a linear scan.

Here is an example command to benchmark the `Approximate KnnDfs` algorithm with `k=10` neighbors and `tol=0.5` on the same tree and queries as the previous example, computing both recall and relative distance error as quality metrics:

```bash
clam-shell \
    --inp-path "../data/string-data/pfam-a/msa-results/10k/trees/tree-pfam_10000-string-levenshtein-span-reduction-factor::sqrt2.tree.bin" \
    --out-dir ../data/string-data/pfam-a/msa-results/10k/bench-results \
    --seed 42 \
    --log-name cakes_bench_pfam_10k \
    cakes \
    --out-fmt yaml \
    --queries-path ../data/string-data/pfam-a/pfam_10000.fasta \
    --num-queries 10 \
    --algorithm "approx-knn-dfs::k=10,tol=0.5" \
    bench \
    --quality-metrics recall \
    --quality-metrics relative-distance-error
```

### PanCakes

TODO...

### Musals

Run the following command to see the options for the `musals` subcommand:

```bash
clam-shell musals --help
```

The `musals` subcommand has the following additional parameters:

- `-f, --out-fmt <OUT_FMT>`: The output format for any reports and results.

The `musals` subcommand has the following subcommands for different use cases:

- `align`: Read the tree built in the `build` step, perform multiple sequence alignment on the sequences, and save the resulting aligned tree to the output directory.
- `evaluate`: Evaluate the quality of the MSA.

#### Align

Run the following command to see the options for the `align` subcommand:

```bash
clam-shell musals align --help
```

The `align` subcommand has the following additional parameters:

- `-c, --cost-matrix <COST_MATRIX>`: The cost matrix to use for multiple sequence alignment.

Here is an example command to perform multiple sequence alignment on the previously built tree from the Pfam dataset using the `affine` cost matrix:

```bash
clam-shell \
    --inp-path "../data/string-data/pfam-a/msa-results/10k/trees/tree-pfam_10000-string-levenshtein-span-reduction-factor::sqrt2.tree.bin" \
    --out-dir ../data/string-data/pfam-a/msa-results/10k/aligned-trees \
    --seed 42 \
    --log-name musals_align_pfam_10k \
    musals \
    align \
    --cost-matrix affine
```

#### Evaluate

Run the following command to see the options for the `evaluate` subcommand:

```bash
clam-shell musals evaluate --help
```

The `evaluate` subcommand has the following additional parameters:

- `-m, --quality-metrics <QUALITY_METRICS>`: One or more quality metrics to compute for the MSA.
- `-n, --num-samples <NUM_SAMPLES>`: An optional parameter to specify the number of sequences to sample from the aligned tree for evaluation. If not specified, all sequences in the aligned tree will be used.
- `-r, --reference-msa <REFERENCE_MSA>`: An optional parameter to specify the path to a reference MSA file in FASTA format. If specified, the quality metrics that require a reference MSA will be computed using the provided reference MSA. If not specified, those quality metrics will be skipped.

Here is an example command to evaluate the quality of the MSA produced in the previous example using all supported quality metrics, sampling 100 sequences from the aligned tree for evaluation, and using a reference MSA from the Pfam dataset:

```bash
clam-shell \
    --inp-path "../data/string-data/pfam-a/msa-results/10k/aligned-trees/aligned-affine-tree-pfam_10000-string-levenshtein-span-reduction-factor::sqrt2.tree.bin" \
    --out-dir ../data/string-data/pfam-a/msa-results/10k/eval-results \
    --seed 42 \
    --log-name musals_eval_pfam_10k \
    musals \
    evaluate \
    --num-samples 100
```

## License

- MIT
