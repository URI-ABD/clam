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
#![allow(unused_imports)]

//! Report the results of an ANN benchmark.

use core::cmp::Ordering;
use std::{path::Path, time::Instant};

use abd_clam::{Cakes, Dataset, PartitionCriteria, VecDataset};
use clap::Parser;
use distances::Number;
use log::info;
use num_format::ToFormattedString;
use serde::{Deserialize, Serialize};

mod ann_datasets;
mod utils;

use crate::{ann_datasets::AnnDatasets, utils::format_f32};

fn main() -> Result<(), String> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = Args::parse();

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

    make_reports(
        &args.input_dir,
        &args.dataset,
        args.use_shards,
        args.tuning_depth,
        args.tuning_k,
        args.ks,
        args.seed,
        &args.output_dir,
    )?;

    Ok(())
}

/// Command line arguments for the replicating the ANN-Benchmarks results for Cakes.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
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
    /// Whether to shard the data set for search.
    #[arg(long)]
    use_shards: bool,
    /// The depth of the tree to use for auto-tuning knn-search.
    #[arg(long, default_value = "10")]
    tuning_depth: usize,
    /// The value of k to use for auto-tuning knn-search.
    #[arg(long, default_value = "10")]
    tuning_k: usize,
    /// Number of nearest neighbors to search for.
    #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "10 100")]
    ks: Vec<usize>,
    /// Seed for the random number generator.
    #[arg(long)]
    seed: Option<u64>,
}

/// Report the results of an ANN benchmark.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn make_reports(
    input_dir: &Path,
    dataset: &str,
    use_shards: bool,
    tuning_depth: usize,
    tuning_k: usize,
    ks: Vec<usize>,
    seed: Option<u64>,
    output_dir: &Path,
) -> Result<(), String> {
    info!("");

    let dataset = AnnDatasets::from_str(dataset)?;
    let metric = dataset.metric()?;
    let [train_data, queries] = dataset.read(input_dir)?;
    info!("Dataset: {}", dataset.name());

    let (cardinality, dimensionality) = (train_data.len(), train_data[0].len());
    info!(
        "Cardinality: {}",
        cardinality.to_formatted_string(&num_format::Locale::en)
    );
    info!(
        "Dimensionality: {}",
        dimensionality.to_formatted_string(&num_format::Locale::en)
    );

    let queries = queries.iter().collect::<Vec<_>>();
    let num_queries = queries.len();
    info!(
        "Number of queries: {}",
        num_queries.to_formatted_string(&num_format::Locale::en)
    );

    let cakes = if use_shards {
        let max_cardinality = if cardinality < 1_000_000 {
            cardinality
        } else if cardinality < 5_000_000 {
            100_000
        } else {
            1_000_000
        };

        let shards = VecDataset::<_, _, bool>::new(
            dataset.name().to_string(),
            train_data,
            metric,
            false,
            None,
        )
        .make_shards(max_cardinality);
        let mut cakes = Cakes::new_randomly_sharded(shards, seed, &PartitionCriteria::default());
        cakes.auto_tune_knn(tuning_k, tuning_depth);
        cakes
    } else {
        let data = VecDataset::new(dataset.name().to_string(), train_data, metric, false, None);
        let mut cakes = Cakes::new(data, seed, &PartitionCriteria::default());
        cakes.auto_tune_knn(tuning_k, tuning_depth);
        cakes
    };

    let shard_sizes = cakes.shard_cardinalities();
    info!(
        "Shard sizes: [{}]",
        shard_sizes
            .iter()
            .map(|s| s.to_formatted_string(&num_format::Locale::en))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let algorithm = cakes.tuned_knn_algorithm();
    info!("Tuned algorithm: {}", algorithm.name());

    for k in ks {
        info!("k: {k}");

        let start = Instant::now();
        let hits = cakes.batch_tuned_knn_search(&queries, k);
        let elapsed = start.elapsed().as_secs_f32();
        let throughput = queries.len().as_f32() / elapsed;
        info!("Throughput: {} QPS", format_f32(throughput));

        let start = Instant::now();
        let linear_hits = cakes.batch_linear_knn_search(&queries, k);
        let linear_elapsed = start.elapsed().as_secs_f32();
        let linear_throughput = queries.len().as_f32() / linear_elapsed;
        info!("Linear throughput: {} QPS", format_f32(linear_throughput));

        let speedup_factor = throughput / linear_throughput;
        info!("Speedup factor: {}", format_f32(speedup_factor));

        let recall = hits
            .into_iter()
            .zip(linear_hits)
            .map(|(mut hits, mut linear_hits)| {
                hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
                let mut hits = hits.into_iter().map(|(_, d)| d).peekable();

                linear_hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
                let mut linear_hits = linear_hits.into_iter().map(|(_, d)| d).peekable();

                let mut num_common = 0;
                while let (Some(&hit), Some(&linear_hit)) = (hits.peek(), linear_hits.peek()) {
                    if (hit - linear_hit).abs() < f32::EPSILON {
                        num_common += 1;
                        hits.next();
                        linear_hits.next();
                    } else if hit < linear_hit {
                        hits.next();
                    } else {
                        linear_hits.next();
                    }
                }
                num_common.as_f32() / k.as_f32()
            })
            .sum::<f32>()
            / queries.len().as_f32();
        info!("Recall: {}", format_f32(recall));

        Report {
            dataset: dataset.name(),
            metric: dataset.metric_name(),
            cardinality,
            dimensionality,
            shard_sizes: shard_sizes.clone(),
            num_queries,
            k,
            tuned_algorithm: algorithm.name(),
            throughput,
            recall,
            linear_throughput,
        }
        .save(output_dir)?;
    }

    Ok(())
}

/// A report of the results of an ANN benchmark.
#[derive(Debug, Serialize, Deserialize)]
struct Report<'a> {
    /// Name of the data set.
    dataset: &'a str,
    /// Name of the distance function.
    metric: &'a str,
    /// Number of data points in the data set.
    cardinality: usize,
    /// Dimensionality of the data set.
    dimensionality: usize,
    /// Sizes of the shards created for `ShardedCakes`.
    shard_sizes: Vec<usize>,
    /// Number of queries used for search.
    num_queries: usize,
    /// Number of nearest neighbors to search for.
    k: usize,
    /// Name of the algorithm used after auto-tuning.
    tuned_algorithm: &'a str,
    /// Throughput of the tuned algorithm.
    throughput: f32,
    /// Throughput of linear search.
    linear_throughput: f32,
    /// Recall of the tuned algorithm.
    recall: f32,
}

impl Report<'_> {
    /// Save the report to a file in the given directory.
    fn save(&self, dir: &Path) -> Result<(), String> {
        let path = dir.join(format!("{}_{}.json", self.dataset, self.k));
        let report = serde_json::to_string_pretty(&self).map_err(|e| e.to_string())?;
        std::fs::write(path, report).map_err(|e| e.to_string())?;
        Ok(())
    }
}
