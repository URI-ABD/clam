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
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_lossless
)]
#![doc = include_str!("../README.md")]

use std::path::PathBuf;

use clap::Parser;

/// Reproducible results for the PANCAKES paper.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
#[allow(clippy::struct_excessive_bools)]
struct Args {
    /// Path to the input file.
    #[arg(short('i'), long)]
    inp_dir: PathBuf,

    /// The dataset to benchmark.
    #[arg(short('d'), long)]
    dataset: bench_utils::RawData,

    /// The number of queries to use for benchmarking.
    #[arg(short('q'), long)]
    num_queries: usize,

    /// Whether to count the number of distance computations during search.
    #[arg(short('c'), long, default_value = "false")]
    count_distance_calls: bool,

    /// The seed for the random number generator.
    #[arg(short('s'), long)]
    seed: Option<u64>,

    /// The maximum time, in seconds, to run each algorithm.
    #[arg(short('t'), long, default_value = "10.0")]
    max_time: f32,

    /// Whether to run ranged search benchmarks.
    #[arg(short('r'), long)]
    ranged_search: bool,

    /// Path to the output directory.
    #[arg(short('o'), long)]
    out_dir: Option<PathBuf>,

    /// Whether to rebuild the trees.
    #[arg(short('w'), long)]
    rebuild_trees: bool,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    println!("Args: {args:?}");

    let log_name = format!("cakes-{}", args.dataset.name());
    let (_guard, log_path) = bench_utils::configure_logger(&log_name)?;
    println!("Log file: {log_path:?}");

    ftlog::info!("{args:?}");

    // Check the input and output directories.
    let inp_dir = args.inp_dir.canonicalize().map_err(|e| e.to_string())?;
    ftlog::info!("Input directory: {inp_dir:?}");

    Ok(())
}
