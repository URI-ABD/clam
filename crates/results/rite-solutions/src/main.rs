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

use abd_clam::{
    cluster::{ParPartition, Partition},
    Ball, Cluster, Dataset,
};
use clap::Parser;
use rayon::prelude::*;

mod data;
mod utils;

/// Reproducible results for the CAKES and panCAKES papers.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the file containing the input data. If None, then the
    /// randomization parameter must be provided.
    #[arg(short('i'), long)]
    inp_path: Option<PathBuf>,

    /// The metric to use.
    #[arg(short('m'), long)]
    metric: data::VecMetric,

    /// The number of randomly generated inliers.
    #[arg(long)]
    num_inliers: Option<usize>,

    /// The dimensionality of the randomly generated inliers.
    #[arg(long)]
    dimensionality: Option<usize>,

    /// Mean of the randomly generated inliers.
    #[arg(long)]
    inlier_mean: Option<f32>,

    /// Standard deviation of the randomly generated inliers.
    #[arg(long)]
    inlier_std: Option<f32>,

    /// The number of randomly generated outliers.
    #[arg(long)]
    num_outliers: usize,

    /// Mean of the randomly generated outliers.
    #[arg(long)]
    outlier_mean: f32,

    /// Standard deviation of the randomly generated outliers.
    #[arg(long)]
    outlier_std: f32,

    /// Random seed.
    #[arg(long)]
    seed: Option<u64>,

    /// Neighborhood size for the `NeighborhoodAware` data structure.
    #[arg(long, default_value = "100")]
    neighborhood_size: usize,

    /// Whether to parallelize the algorithms.
    #[arg(long, default_value = "true")]
    parallelize: bool,

    /// Path to the output directory. This is where the results and logs
    /// will be saved.
    #[arg(short('o'), long)]
    out_dir: PathBuf,
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    let out_dir = args.out_dir.canonicalize().map_err(|e| e.to_string())?;
    if !out_dir.is_dir() {
        return Err(format!("{out_dir:?} is not a directory"));
    }
    if !out_dir.exists() {
        return Err(format!("{out_dir:?} does not exist"));
    }

    let log_path = out_dir.join("rite-logs.log");
    println!("Log file: {log_path:?}");

    let _guard = utils::configure_logger(&log_path)?;
    ftlog::info!("Args: {args:?}");

    let data = data::read_or_generate(
        args.inp_path,
        args.num_inliers,
        args.dimensionality,
        args.inlier_mean,
        args.inlier_std,
        args.seed,
    )?;

    let metric = args.metric.metric();
    let criteria = |c: &Ball<_>| c.cardinality() > 1;
    let root = if args.parallelize {
        Ball::par_new_tree(&data, &metric, &criteria, args.seed)
    } else {
        Ball::new_tree(&data, &metric, &criteria, args.seed)
    };

    let data = if args.parallelize {
        data::NeighborhoodAware::par_new(&data, &metric, &root, args.neighborhood_size)
    } else {
        data::NeighborhoodAware::new(&data, &metric, &root, args.neighborhood_size)
    };

    let dim = data.dimensionality_hint().0;
    let outliers = data::gen_random(args.outlier_mean, args.outlier_std, args.num_outliers, dim, args.seed);

    let results = if args.parallelize {
        outliers
            .par_iter()
            .map(|outlier| data.is_outlier(&metric, &root, outlier, 0.5))
            .collect::<Vec<_>>()
    } else {
        outliers
            .iter()
            .map(|outlier| data.is_outlier(&metric, &root, outlier, 0.5))
            .collect()
    };
    let results = results.into_iter().enumerate().collect::<Vec<_>>();

    ftlog::info!("Results: {results:?}");

    Ok(())
}
