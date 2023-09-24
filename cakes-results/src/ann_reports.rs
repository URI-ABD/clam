//! Report the results of an ANN benchmark.

use core::cmp::Ordering;
use std::{path::Path, time::Instant};

use abd_clam::{Cakes, Dataset, PartitionCriteria, ShardedCakes, VecDataset};
use distances::Number;
use log::info;
use num_format::ToFormattedString;
use serde::{Deserialize, Serialize};

use crate::ann_datasets::AnnDatasets;

/// Report the results of an ANN benchmark.
pub fn make_reports(
    input_dir: &Path,
    dataset: &str,
    tuning_depth: usize,
    tuning_k: usize,
    k: usize,
    seed: Option<u64>,
    output_dir: &Path,
) -> Result<(), String> {
    let dataset = AnnDatasets::from_str(dataset)?;
    let metric = dataset.metric();
    let [train_data, queries] = dataset.read(input_dir)?;
    info!("Dataset: {}", dataset.name());
    info!("k: {k}");

    let train_data = train_data.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let (cardinality, dimensionality) = (train_data.len(), train_data[0].len());
    info!(
        "Cardinality: {}",
        cardinality.to_formatted_string(&num_format::Locale::en)
    );
    info!(
        "Dimensionality: {}",
        dimensionality.to_formatted_string(&num_format::Locale::en)
    );

    let queries = queries.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let num_queries = queries.len();
    info!(
        "Number of queries: {}",
        num_queries.to_formatted_string(&num_format::Locale::en)
    );

    let max_cardinality = if cardinality < 1_000_000 {
        cardinality
    } else if cardinality < 5_000_000 {
        100_000
    } else {
        1_000_000
    };

    let data_shards = VecDataset::new(dataset.name().to_string(), train_data, metric, false)
        .make_shards(max_cardinality);
    let shards = data_shards
        .into_iter()
        .map(|d| Cakes::new(d, seed, PartitionCriteria::default()))
        .collect::<Vec<_>>();
    let cakes = ShardedCakes::new(shards);

    let shard_sizes = cakes.shard_cardinalities();
    info!(
        "Shard sizes: [{}]",
        shard_sizes
            .iter()
            .map(|s| s.to_formatted_string(&num_format::Locale::en))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let cakes = cakes.auto_tune(tuning_depth, tuning_k);

    let algorithm = cakes.best_knn_algorithm();
    info!("Tuned algorithm: {}", algorithm.name());

    let start = Instant::now();
    let hits = cakes.batch_knn_search(&queries, k);
    let elapsed = start.elapsed().as_secs_f32();
    let throughput = queries.len().as_f32() / elapsed;
    info!("Throughput: {} QPS", format_f32(throughput));

    let start = Instant::now();
    let linear_hits = cakes.batch_linear_knn(&queries, k);
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
        cardinality,
        dimensionality,
        shard_sizes,
        num_queries,
        k,
        tuned_algorithm: algorithm.name(),
        throughput,
        recall,
        linear_throughput,
    }
    .save(output_dir)?;

    Ok(())
}

/// Format a `f32` as a string with 6 digits of precision and separators.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn format_f32(x: f32) -> String {
    let trunc = x.trunc() as u32;
    let fract = (x.fract() * 10f32.powi(6)).round() as u32;

    let trunc = trunc.to_formatted_string(&num_format::Locale::en);

    #[allow(clippy::unwrap_used)]
    let fract = fract.to_formatted_string(
        &num_format::CustomFormat::builder()
            .grouping(num_format::Grouping::Standard)
            .separator("_")
            .build()
            .unwrap(),
    );

    format!("{trunc}.{fract}")
}

/// A report of the results of an ANN benchmark.
#[derive(Debug, Serialize, Deserialize)]
struct Report<'a> {
    /// Name of the data set.
    dataset: &'a str,
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
