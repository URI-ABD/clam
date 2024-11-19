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

use clap::Parser;

mod benchmarks;
mod data_types;
mod distance_functions;
mod utils;

/// Reproducible benchmarks for CLAM.
#[derive(Parser, Debug)]
#[command(about)]
struct Args {
    /// Benchmarks to run.
    #[command(subcommand)]
    benchmark: benchmarks::Benchmark,

    /// Path to the input dataset.
    #[arg(short('i'), long)]
    inp_path: PathBuf,

    /// The type of the input dataset.
    #[arg(long)]
    data_type: data_types::Dataset,

    /// The type of values in the input dataset.
    #[arg(long)]
    inp_type: data_types::Units,

    /// The type of distance values.
    #[arg(long)]
    dist_type: data_types::Units,

    /// Distance function to use for building the tree.
    #[arg(short('f'), long)]
    distance_function: distance_functions::DistanceFunctions,

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

fn main() -> Result<(), String> {
    let args = Args::parse();
    println!("Args: {args:?}");

    let (_guard, log_path) = utils::configure_logger(&args)?;
    println!("Log path: {log_path:?}");

    ftlog::info!("Args: {args:?}");

    match args.benchmark {
        benchmarks::Benchmark::Search { .. } => benchmarks::search(&args),
    }
}
