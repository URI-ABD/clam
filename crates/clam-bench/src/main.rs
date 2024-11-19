#![deny(clippy::correctness)]
#![warn(
    missing_docs,
    clippy::all,
    clippy::suspicious,
    clippy::style,
    clippy::complexity,
    clippy::perf,
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_docs_in_private_items,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_lossless
)]
#![doc = include_str!("../README.md")]

use std::path::PathBuf;

use clap::{Parser, Subcommand};

mod utils;

/// Reproducible benchmarks for CLAM.
#[derive(Parser, Debug)]
#[command(about)]
struct Args {
    /// Benchmarks to run.
    #[command(subcommand)]
    benchmark: Benchmark,

    /// Path to the input dataset.
    #[arg(short('i'), long)]
    inp_path: PathBuf,

    /// The type of the input dataset.
    #[arg(long)]
    data_type: DatasetType,

    /// The type of values in the input dataset.
    #[arg(long)]
    inp_type: DType,

    /// The type of distance values.
    #[arg(long)]
    dist_type: DType,

    /// Distance function to use for building the tree.
    #[arg(short('f'), long)]
    distance_function: DistanceFunction,

    /// Path to the output directory.
    #[arg(short('o'), long)]
    out_dir: PathBuf,

    /// The maximum depth of the tree. By default, the tree is grown until all
    /// leaf nodes are singletons.
    #[arg(long)]
    max_depth: Option<usize>,

    /// The minimum cardinality of a leaf node. By default, the tree is grown
    /// until all leaf nodes are singletons.
    #[arg(long)]
    min_cardinality: Option<usize>,

    /// Number of threads to use. By default, all available threads are used.
    #[arg(long)]
    num_threads: Option<usize>,
}

/// Benchmarks to use.
#[derive(Subcommand, Debug)]
enum Benchmark {
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
enum TreeTypes {
    /// Tree with permuted data to speed up search.
    #[clap(name = "permuted")]
    Permuted,
    /// Compressed tree.
    #[clap(name = "compressed")]
    Compressed,
}

/// The data types that can be used for the input dataset.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum DatasetType {
    /// 2d-array in the `.npy` format.
    #[clap(name = "vectors")]
    Vectors,
    /// Set data. A plain-text file in which each row is a space-separated set
    /// of elements enumerated as unsigned integers.
    #[clap(name = "sets")]
    Sets,
    /// Sequences in a FASTA file.
    #[clap(name = "fasta")]
    Fasta,
}

/// The data types that can be used.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum DType {
    /// f32.
    #[clap(name = "f32")]
    F32,
    /// f64.
    #[clap(name = "f64")]
    F64,
    /// i8.
    #[clap(name = "i8")]
    I8,
    /// i16.
    #[clap(name = "i16")]
    I16,
    /// i32.
    #[clap(name = "i32")]
    I32,
    /// i64.
    #[clap(name = "i64")]
    I64,
    /// isize.
    #[clap(name = "isize")]
    Isize,
    /// u8.
    #[clap(name = "u8")]
    U8,
    /// u16.
    #[clap(name = "u16")]
    U16,
    /// u32.
    #[clap(name = "u32")]
    U32,
    /// u64.
    #[clap(name = "u64")]
    U64,
    /// usize.
    #[clap(name = "usize")]
    Usize,
    /// bool.
    #[clap(name = "bool")]
    Bool,
    /// Char.
    #[clap(name = "char")]
    Char,
    /// Strings.
    #[clap(name = "string")]
    String,
}

/// The distance functions that can be used for nearest neighbors search.
#[derive(clap::ValueEnum, Debug, Clone)]
pub enum DistanceFunction {
    /// Euclidean distance.
    #[clap(name = "euclidean")]
    Euclidean,
    /// Manhattan distance.
    #[clap(name = "manhattan")]
    Manhattan,
    /// Cosine distance.
    #[clap(name = "cosine")]
    Cosine,
    /// Hamming distance.
    #[clap(name = "hamming")]
    Hamming,
    /// Levenshtein edit distance.
    #[clap(name = "levenshtein")]
    Levenshtein,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let (_guard, log_path) = utils::configure_logger(&args)?;
    println!("Log path: {log_path:?}");

    ftlog::info!("Args: {args:?}");

    Ok(())
}
