use core::cmp::Ordering;
use std::{path::Path, time::Instant};

use abd_clam::{Cakes, Dataset, PartitionCriteria, ShardedCakes, VecDataset};
use distances::Number;
use ndarray::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

enum Metrics {
    Cosine,
    Euclidean,
}

impl Metrics {
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(Metrics::Cosine),
            "euclidean" => Ok(Metrics::Euclidean),
            _ => Err(format!("Unknown metric: {}", s)),
        }
    }

    fn distance(&self) -> fn(&[f32], &[f32]) -> f32 {
        match self {
            Metrics::Cosine => distances::vectors::cosine,
            Metrics::Euclidean => distances::simd::euclidean_f32,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Report<'a> {
    data_name: &'a str,
    metric_name: &'a str,
    cardinality: usize,
    dimensionality: usize,
    shard_sizes: Vec<usize>,
    num_queries: usize,
    k: usize,
    algorithm: &'a str,
    throughput: f32,
    linear_throughput: f32,
    recall: f32,
}

impl Report<'_> {
    fn save(&self, dir: &Path) -> Result<(), String> {
        let path = dir.join(format!("{}_{}.json", self.data_name, self.metric_name));
        let report = serde_json::to_string_pretty(&self).map_err(|e| e.to_string())?;
        std::fs::write(path, report).map_err(|e| e.to_string())?;
        Ok(())
    }
}

pub fn make_reports(
    data_dir: &Path,
    data_name: &str,
    metric_name: &str,
    tuning_depth: usize,
    tuning_k: usize,
    k: usize,
    output_dir: &Path,
) -> Result<(), String> {
    let metric = Metrics::from_str(metric_name)?.distance();
    let [train_data, queries] = read_search_data(data_dir, data_name)?;
    println!("dataset: {data_name}, metric: {metric_name}");

    let train_data = train_data.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let (cardinality, dimensionality) = (train_data.len(), train_data[0].len());
    println!("cardinality: {cardinality}, dimensionality: {dimensionality}");

    let queries = queries.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let num_queries = queries.len();
    println!("num_queries: {num_queries}");

    let max_cardinality = if cardinality > 1_000_000 {
        cardinality / 10
    } else {
        cardinality
    };

    let data_shards = VecDataset::new(data_name.to_string(), train_data, metric, false)
        .make_shards(max_cardinality);
    let shards = data_shards
        .into_iter()
        .map(|d| {
            let threshold = d.cardinality().as_f64().log2().ceil() as usize;
            let criteria = PartitionCriteria::new(true).with_min_cardinality(threshold);
            Cakes::new(d, None, criteria)
        })
        .collect::<Vec<_>>();
    let cakes = ShardedCakes::new(shards).auto_tune(tuning_k, tuning_depth);

    let shard_sizes = cakes.shard_cardinalities();
    println!("shard_sizes: {shard_sizes:?}");

    let algorithm = cakes.best_knn_algorithm();
    println!("algorithm: {}", algorithm.name());

    let start = Instant::now();
    let hits = cakes.batch_knn_search(&queries, k);
    let elapsed = start.elapsed().as_secs_f32();
    let throughput = queries.len().as_f32() / elapsed;
    println!("throughput: {throughput}");

    let start = Instant::now();
    let linear_hits = cakes.batch_linear_knn(&queries, k);
    let linear_elapsed = start.elapsed().as_secs_f32();
    let linear_throughput = queries.len().as_f32() / linear_elapsed;
    println!("linear_throughput: {linear_throughput}");

    let speedup_factor = throughput / linear_throughput;
    println!("speedup_factor: {speedup_factor}");

    let recall = hits
        .into_par_iter()
        .zip(linear_hits.into_par_iter())
        .map(|(mut hits, mut linear_hits)| {
            hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
            let mut hits = hits.into_iter().map(|(_, d)| d).peekable();

            linear_hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
            let mut linear_hits = linear_hits.into_iter().map(|(_, d)| d).peekable();

            let mut num_common = 0;
            while let (Some(&hit), Some(&linear_hit)) = (hits.peek(), linear_hits.peek()) {
                if hit < linear_hit {
                    hits.next();
                } else if hit > linear_hit {
                    linear_hits.next();
                } else {
                    num_common += 1;
                    hits.next();
                    linear_hits.next();
                }
            }
            num_common.as_f32() / k.as_f32()
        })
        .sum::<f32>()
        / queries.len().as_f32();
    println!("recall: {recall}");

    Report {
        data_name,
        metric_name,
        cardinality,
        dimensionality,
        shard_sizes,
        num_queries,
        k,
        algorithm: algorithm.name(),
        throughput,
        recall,
        linear_throughput,
    }
    .save(output_dir)?;

    Ok(())
}

fn read_search_data(dir: &Path, name: &str) -> Result<[Vec<Vec<f32>>; 2], String> {
    let train_path = dir.join(format!("{}-train.npy", name));
    let train_data = read_npy(&train_path)?;

    let test_path = dir.join(format!("{}-test.npy", name));
    let test_data = read_npy(&test_path)?;

    Ok([train_data, test_data])
}

fn read_npy(path: &Path) -> Result<Vec<Vec<f32>>, String> {
    let data: Array2<f32> = ndarray_npy::read_npy(path).map_err(|error| {
        format!(
            "Error: Failed to read your dataset at {}. {:}",
            path.to_str().unwrap(),
            error
        )
    })?;

    Ok(data.outer_iter().map(|row| row.to_vec()).collect())
}
