use std::{path::Path, time::Instant};

use abd_clam::{Cakes, Dataset, PartitionCriteria, ShardedCakes, VecDataset};
use distances::Number;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::ann_readers;

pub enum Metrics {
    Cosine,
    Euclidean,
}

impl Metrics {
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "cosine" | "angular" => Ok(Metrics::Cosine),
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
    fn save(&self, directory: &Path) -> Result<(), String> {
        let path = directory.join(format!(
            "{}_{}_{}_{}_{}.json",
            self.data_name,
            self.metric_name,
            self.k,
            self.algorithm,
            self.shard_sizes.len()
        ));
        let report = serde_json::to_string(&self).map_err(|e| e.to_string())?;
        std::fs::write(path, report).map_err(|e| e.to_string())?;
        Ok(())
    }
}

pub fn make_reports() -> Result<(), String> {
    let reports_dir = {
        let mut dir = std::env::current_dir().map_err(|e| e.to_string())?;
        dir.push("reports");
        if !dir.exists() {
            std::fs::create_dir(&dir).map_err(|e| e.to_string())?;
        }
        dir
    };

    for &(data_name, metric_name) in ann_readers::DATASETS {
        if ["kosarak", "nytimes"].contains(&data_name) {
            continue;
        }

        let metric = if let Ok(metric) = Metrics::from_str(metric_name) {
            metric.distance()
        } else {
            continue;
        };
        println!("dataset: {data_name}, metric: {metric_name}");

        let (train_data, queries) = ann_readers::read_search_data(data_name)?;
        let (cardinality, dimensionality) = (train_data.len(), train_data[0].len());
        println!("cardinality: {cardinality}, dimensionality: {dimensionality}");

        let train_data = train_data.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let queries = queries.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let num_queries = queries.len();
        println!("num_queries: {num_queries}");

        let max_cardinality = if cardinality > 1_000_000 {
            cardinality / 10
        } else {
            cardinality
        };

        let threshold = max_cardinality.as_f64().log2().ceil() as usize;
        let data_shards = VecDataset::new(data_name.to_string(), train_data, metric, false)
            .make_shards(max_cardinality);
        let shards = data_shards
            .into_iter()
            .map(|d| {
                let criteria = PartitionCriteria::new(true).with_min_cardinality(threshold);
                Cakes::new(d, None, criteria)
            })
            .collect::<Vec<_>>();

        let sampling_depth = 10;
        let cakes = ShardedCakes::new(shards).auto_tune(10, sampling_depth);

        for k in (1..3).map(|v| 10usize.pow(v)) {
            println!("\tk: {k}");

            let algorithm = cakes.best_knn_algorithm();
            println!("\t\talgorithm: {}", algorithm.name());

            let start = Instant::now();
            let hits = cakes.batch_knn_search(&queries, k);
            let elapsed = start.elapsed().as_secs_f32();
            let throughput = queries.len().as_f32() / elapsed;

            let start = Instant::now();
            let linear_hits = cakes.batch_linear_knn(&queries, k);
            let linear_elapsed = start.elapsed().as_secs_f32();
            let linear_throughput = queries.len().as_f32() / linear_elapsed;

            let recall = hits
                .into_par_iter()
                .zip(linear_hits.into_par_iter())
                .map(|(hits, linear_hits)| {
                    let mut hits = hits.into_iter().map(|(_, d)| d).peekable();
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

            Report {
                data_name,
                metric_name,
                cardinality,
                dimensionality,
                shard_sizes: cakes.shard_cardinalities(),
                num_queries,
                k,
                algorithm: algorithm.name(),
                throughput,
                recall,
                linear_throughput,
            }
            .save(&reports_dir)?;
        }

        drop(cakes);
    }

    Ok(())
}
