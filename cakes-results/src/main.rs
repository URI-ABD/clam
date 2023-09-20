#![doc = include_str!("../README.md")]

use std::path::PathBuf;

use clap::Parser;

mod ann_reports;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct AnnArgs {
    /// Path to the directory with the data sets. The directory should contain
    /// the hdf5 files downloaded from the ann-benchmarks repository.
    #[arg(long)]
    data_dir: PathBuf,
    /// Name of the data set to process. `data_dir` should contain two files
    /// named `{name}-train.npy` and `{name}-test.npy`. The train file
    /// contains the data to be indexed for search, and the test file contains
    /// the queries to be searched for.
    #[arg(long)]
    name: String,
    /// Name (case-insensitive) of the metric to use. Possible values are `cosine` and `euclidean`.
    #[arg(long)]
    metric: String,
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
    /// Output directory for the report.
    #[arg(long)]
    output_dir: PathBuf,
}

fn main() -> Result<(), String> {
    let args = AnnArgs::parse();

    // Check that the data set exists.
    let data_paths = [
        args.data_dir.join(format!("{}-train.npy", args.name)),
        args.data_dir.join(format!("{}-test.npy", args.name)),
    ];
    for path in &data_paths {
        if !path.exists() {
            return Err(format!("File {:?} does not exist.", path));
        }
    }

    // Check that the metric is valid.
    match args.metric.to_lowercase().as_str() {
        "cosine" | "euclidean" => {}
        _ => return Err(format!("Unknown metric: {}", args.metric)),
    }

    // Check that the output directory exists.
    if !args.output_dir.exists() {
        return Err(format!(
            "Output directory {:?} does not exist.",
            args.output_dir
        ));
    }

    ann_reports::make_reports(
        &args.data_dir,
        &args.name,
        &args.metric,
        args.tuning_depth,
        args.tuning_k,
        args.k,
        &args.output_dir,
    )?;

    Ok(())
}
