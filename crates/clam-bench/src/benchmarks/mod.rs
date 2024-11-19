//! All benchmark types.

mod search;

use clap::Subcommand;
pub use search::search;

/// Benchmarks to use.
#[derive(Subcommand, Debug)]
pub enum Benchmark {
    /// Benchmarks for nearest neighbors search.
    Search {
        /// The kinds of trees to build in addition to the default tree.
        #[arg(short('t'), long, default_value = "permuted")]
        tree_types: Option<Vec<TreeTypes>>,
        /// Fractions of the root radius at which to run the benchmarks. All
        /// values must be non-negative.
        #[arg(short('r'), long, default_value = "0.001,0.005,0.01,0.05,0.1,0.5")]
        radii: Vec<f32>,
        /// Values of k to use.
        #[arg(short('k'), long, default_value = "1,10,100,1000")]
        ks: Vec<usize>,
        /// Number of queries to use.
        #[arg(short('q'), long, default_value = "1000")]
        num_queries: usize,
    },
}

/// The kinds of trees to build in addition to the default tree.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum TreeTypes {
    /// Tree with permuted data to speed up search.
    #[clap(name = "permuted")]
    Permuted,
    /// Compressed tree. Compression is only available with the `hamming`,
    /// `jaccard` and `levenshtein` distance functions.
    #[clap(name = "compressed")]
    Compressed,
}
