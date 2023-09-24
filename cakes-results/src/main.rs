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

use clap::Parser;

mod ann_datasets;
mod ann_reports;

/// Command line arguments for the replicating the ANN-Benchmarks results for Cakes.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct AnnArgs {
    /// Path to the directory with the data sets. The directory should contain
    /// the hdf5 files downloaded from the ann-benchmarks repository.
    #[arg(long)]
    input_dir: std::path::PathBuf,
    /// Output directory for the report.
    #[arg(long)]
    output_dir: std::path::PathBuf,
    /// Name of the data set to process. `data_dir` should contain two files
    /// named `{name}-train.npy` and `{name}-test.npy`. The train file
    /// contains the data to be indexed for search, and the test file contains
    /// the queries to be searched for.
    #[arg(long)]
    dataset: String,
    /// The depth of the tree to use for auto-tuning knn-search.
    #[arg(long, default_value = "10")]
    tuning_depth: usize,
    /// The value of k to use for auto-tuning knn-search.
    #[arg(long, default_value = "10")]
    tuning_k: usize,
    /// Number of nearest neighbors to search for.
    #[arg(long, default_value = "10")]
    k: usize,
    /// Seed for the random number generator.
    #[arg(long)]
    seed: Option<u64>,
}

fn main() -> Result<(), String> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = AnnArgs::parse();

    // Check that the data set exists.
    let data_paths = [
        args.input_dir.join(format!("{}-train.npy", args.dataset)),
        args.input_dir.join(format!("{}-test.npy", args.dataset)),
    ];
    for path in &data_paths {
        if !path.exists() {
            return Err(format!("File {path:?} does not exist."));
        }
    }

    // Check that the output directory exists.
    if !args.output_dir.exists() {
        return Err(format!(
            "Output directory {:?} does not exist.",
            args.output_dir
        ));
    }

    ann_reports::make_reports(
        &args.input_dir,
        &args.dataset,
        args.tuning_depth,
        args.tuning_k,
        args.k,
        args.seed,
        &args.output_dir,
    )?;

    Ok(())
}
