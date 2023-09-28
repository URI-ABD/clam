//! Benchmarks for knn-search when the size of the data set is scaled.

use std::{path::Path, time::Instant};

use abd_clam::{knn, Cakes, PartitionCriteria, VecDataset};
use clap::Parser;
use distances::Number;
use log::{error, info};
use num_format::ToFormattedString;
use serde::{Deserialize, Serialize};
use symagen::augmentation;

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
        &args.output_dir,
        &args.dataset,
        args.seed,
        args.max_scale,
        args.error_rate,
        &args.ks,
        args.max_memory,
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
    /// Seed for the random number generator.
    #[arg(long)]
    seed: Option<u64>,
    /// Maximum scaling factor. The data set will be scaled by factors of
    /// `2 ^ i` for `i` in `0..=max_scale`.
    #[arg(long, default_value = "16")]
    max_scale: u32,
    /// Error rate used for scaling.
    #[arg(long, default_value = "0.01")]
    error_rate: f32,
    /// Number of nearest neighbors to search for.
    #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "10 100")]
    ks: Vec<usize>,
    /// Maximum memory usage (in gigabytes) for the scaled data sets.
    #[arg(long, default_value = "256")]
    max_memory: usize,
}

/// Report the results of a scaling benchmark.
#[allow(clippy::too_many_arguments)]
pub fn make_reports(
    input_dir: &Path,
    output_dir: &Path,
    dataset: &str,
    seed: Option<u64>,
    max_scale: u32,
    error_rate: f32,
    ks: &[usize],
    max_memory: usize,
) -> Result<(), String> {
    let dataset = AnnDatasets::from_str(dataset)?;
    let metric = dataset.metric()?;
    let [train_data, queries] = dataset.read(input_dir)?;

    info!("Dataset: {}", dataset.name());

    let base_cardinality = train_data.len();
    info!(
        "Base cardinality: {}",
        base_cardinality.to_formatted_string(&num_format::Locale::en)
    );

    let dimensionality = train_data[0].len();
    info!(
        "Dimensionality: {}",
        dimensionality.to_formatted_string(&num_format::Locale::en)
    );

    let queries = queries
        .iter()
        .take(100)
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let num_queries = queries.len();
    info!(
        "Number of queries: {}",
        num_queries.to_formatted_string(&num_format::Locale::en)
    );

    let csv_name = format!("{}_{}.csv", dataset.name(), (error_rate * 100.) as usize);

    let report = Report {
        dataset: dataset.name(),
        base_cardinality,
        dimensionality,
        num_queries,
        error_rate,
        ks: ks.to_vec(),
        csv_name: &csv_name,
    };
    report.save(output_dir)?;

    let csv_path = output_dir.join(&csv_name);
    if csv_path.exists() {
        std::fs::remove_file(&csv_path).map_err(|e| e.to_string())?;
    }

    let mut csv_writer = csv::Writer::from_path(csv_path).map_err(|e| e.to_string())?;
    csv_writer
        .write_record(["scale", "build_time", "algorithm", "k", "throughput"])
        .map_err(|e| e.to_string())?;

    for multiplier in (0..=max_scale).map(|s| 2_usize.pow(s)) {
        info!("");
        info!("Scaling data by a factor of {}.", multiplier);
        info!("Error rate: {}", error_rate);

        let data = if multiplier == 1 {
            train_data.clone()
        } else {
            // If memory cost would be too high, continue to next scale.
            let memory_cost = memory_cost(base_cardinality * multiplier, dimensionality);
            if memory_cost > max_memory * 1024 * 1024 * 1024 {
                error!("Memory cost would be over 256G. Skipping scale {multiplier}.");
                continue;
            }

            augmentation::augment_data(&train_data, multiplier - 1, error_rate)
        };

        let data = data.iter().map(Vec::as_slice).collect::<Vec<_>>();

        let cardinality = data.len();
        info!(
            "Scaled cardinality: {}",
            cardinality.to_formatted_string(&num_format::Locale::en)
        );

        let data_name = format!("{}-{}", dataset.name(), multiplier + 1);
        let data = VecDataset::new(data_name, data, metric, false);
        let criteria = PartitionCriteria::default();

        let start = Instant::now();
        let cakes = Cakes::new(data, seed, criteria);
        let cakes_time = start.elapsed().as_secs_f32();
        info!("Cakes tree-building time: {:.3e} s", cakes_time);

        for algorithm in knn::Algorithm::variants() {
            info!("Algorithm: {}", algorithm.name());

            for &k in ks {
                info!("k: {}", k);

                let start = Instant::now();
                let hits = cakes.batch_knn_search(&queries, k, *algorithm);
                let elapsed = start.elapsed().as_secs_f32();
                let throughput = num_queries.as_f32() / elapsed;
                info!("Throughput: {} QPS", format_f32(throughput));

                csv_writer
                    .write_record(&[
                        (multiplier + 1).to_string(),
                        cakes_time.to_string(),
                        algorithm.name().to_string(),
                        k.to_string(),
                        throughput.to_string(),
                    ])
                    .map_err(|e| e.to_string())?;

                // write what we have so far to csv
                csv_writer.flush().map_err(|e| e.to_string())?;

                let misses = hits
                    .into_iter()
                    .map(|h| h.len())
                    .filter(|&h| h != k)
                    .collect::<Vec<_>>();
                if !misses.is_empty() {
                    let &min_hits = misses.iter().min().unwrap();
                    let &max_hits = misses.iter().max().unwrap();
                    error!(
                        "{} queries did not get enough hits. Expected {}, got between {} and {}.",
                        misses.len(),
                        k,
                        min_hits,
                        max_hits
                    )
                }
            }
        }
    }

    // // finalize csv
    // csv_writer.flush().map_err(|e| e.to_string())?;

    Ok(())
}

fn memory_cost(cardinality: usize, dimensionality: usize) -> usize {
    cardinality * dimensionality * std::mem::size_of::<f32>()
}

/// A report of the results of a scaling benchmark.
#[derive(Debug, Serialize, Deserialize)]
struct Report<'a> {
    /// Name of the data set.
    dataset: &'a str,
    /// Cardinality of the real data set.
    base_cardinality: usize,
    /// Dimensionality of the data set.
    dimensionality: usize,
    /// Number of queries.
    num_queries: usize,
    /// Error rate used for scaling.
    error_rate: f32,
    /// Values of k used for knn-search.
    ks: Vec<usize>,
    /// Csv name
    csv_name: &'a str,
}

impl Report<'_> {
    /// Save the report to a file in the given directory.
    fn save(&self, dir: &Path) -> Result<(), String> {
        let e = (self.error_rate * 100.) as usize;
        let path = dir.join(format!("{}_{}.json", self.dataset, e));
        let report = serde_json::to_string_pretty(&self).map_err(|e| e.to_string())?;
        std::fs::write(path, report).map_err(|e| e.to_string())?;
        Ok(())
    }
}
