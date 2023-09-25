//! Benchmarks for knn-search when the size of the data set is scaled.

use std::{path::Path, time::Instant};

use abd_clam::{knn, Cakes, PartitionCriteria, VecDataset};
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
        &args.output_dir,
        &args.dataset,
        args.seed,
        &args.scales,
        args.error_rate,
        &args.ks,
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
    /// Dataset scaling factors.
    #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "0 1 9 99")]
    scales: Vec<usize>,
    /// Error rate used for scaling.
    #[arg(long, default_value = "0.01")]
    error_rate: f32,
    /// Number of nearest neighbors to search for.
    #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "10 100")]
    ks: Vec<usize>,
}

/// Report the results of a scaling benchmark.
pub fn make_reports(
    input_dir: &Path,
    output_dir: &Path,
    dataset: &str,
    seed: Option<u64>,
    scales: &[usize],
    error_rate: f32,
    ks: &[usize],
) -> Result<(), String> {
    let dataset = AnnDatasets::from_str(dataset)?;
    let metric = dataset.metric();
    let [train_data, queries] = dataset.read(input_dir)?;
    info!("Dataset: {}", dataset.name());

    let train_data = train_data.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let dimensionality = train_data[0].len();

    let queries = queries.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let num_queries = queries.len();
    info!(
        "Number of queries: {}",
        num_queries.to_formatted_string(&num_format::Locale::en)
    );

    let mut scale_times = Vec::new();
    for &scale in scales {
        let data = if scale == 0 {
            train_data.clone()
        } else {
            info!("Scaling data by a factor of {}.", scale + 1);
            info!("Error rate: {}", error_rate);

            // TODO: Use function from symagen.
            let mut data = Vec::with_capacity(train_data.len() * (scale + 1));
            for _ in 0..=scale {
                data.extend_from_slice(&train_data);
            }

            data
        };

        let cardinality = data.len();
        info!(
            "Cardinality: {}",
            cardinality.to_formatted_string(&num_format::Locale::en)
        );
        info!(
            "Dimensionality: {}",
            dimensionality.to_formatted_string(&num_format::Locale::en)
        );

        let data_name = format!("{}-{}", dataset.name(), scale + 1);
        let data = VecDataset::new(data_name, data, metric, false);
        let criteria = PartitionCriteria::default();

        let start = Instant::now();
        let cakes = Cakes::new(data, seed, criteria);
        let cakes_time = start.elapsed().as_secs_f32();
        info!("Cakes tree-building time: {:.3e} s", cakes_time);

        let mut algo_times = Vec::new();
        for algorithm in knn::Algorithm::variants() {
            info!("Algorithm: {}", algorithm.name());

            let mut k_throughput = Vec::new();
            for &k in ks {
                info!("k: {}", k);

                let start = Instant::now();
                let hits = cakes.batch_knn_search(&queries, k, *algorithm);
                let elapsed = start.elapsed().as_secs_f32();
                let throughput = num_queries.as_f32() / elapsed;
                info!("Throughput: {} QPS", format_f32(throughput));

                k_throughput.push((k, throughput));

                assert_eq!(hits.len(), num_queries);
                for h in hits {
                    assert_eq!(h.len(), k);
                }
            }

            algo_times.push((algorithm.name(), k_throughput));
        }

        scale_times.push((scale, cakes_time, algo_times));
    }

    let report = Report {
        dataset: dataset.name(),
        base_cardinality: train_data.len(),
        dimensionality,
        num_queries,
        scales: scales.to_vec(),
        error_rate,
        ks: ks.to_vec(),
        throughput: scale_times,
    };
    report.save(output_dir)?;

    Ok(())
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
    /// Dataset scaling factors.
    scales: Vec<usize>,
    /// Error rate used for scaling.
    error_rate: f32,
    /// Values of k used for knn-search.
    ks: Vec<usize>,
    /// Throughput measurements for each scale, algorithm and k.
    ///
    /// The outer vector is of (scale, cakes_build_time, `algo_times`)
    /// `algo_times` is a vector of (algorithm_name, `k_throughput`)
    /// `k_throughput` is a vector of (k, throughput)
    #[allow(clippy::type_complexity)]
    throughput: Vec<(usize, f32, Vec<(&'a str, Vec<(usize, f32)>)>)>,
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
