use criterion::*;
use distances::Number;

use abd_clam::{knn, Cakes, Dataset, PartitionCriteria, ShardedCakes, VecDataset};

mod utils;

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

fn cakes(c: &mut Criterion) {
    let seed = 42;
    let num_queries = 50;

    for &(data_name, metric_name) in utils::DATASETS {
        if data_name != "glove-25" {
            continue;
        }

        let metric = if let Ok(metric) = Metrics::from_str(metric_name) {
            metric.distance()
        } else {
            continue;
        };

        let (train_data, queries) = utils::read_search_data(data_name).unwrap();
        let (cardinality, _dimensionality) = (train_data.len(), train_data[0].len());

        let train_data = train_data.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let queries = queries.iter().take(num_queries).map(Vec::as_slice).collect::<Vec<_>>();

        let mut group = c.benchmark_group(format!("sharded-{data_name}-{metric_name}"));
        group
            .sample_size(10)
            .sampling_mode(SamplingMode::Flat)
            .throughput(Throughput::Elements(num_queries as u64))
            .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

        let data = VecDataset::new("full".to_string(), train_data.clone(), metric, false);
        let min_cardinality = data.cardinality().as_f64().log2().ceil() as usize;
        let criteria = PartitionCriteria::new(true).with_min_cardinality(min_cardinality);

        let cakes = Cakes::new(data, Some(seed), criteria);
        // Run benchmarks on a single shard
        for k in (0..3).map(|i| 10usize.pow(i)) {
            for &algorithm in knn::Algorithm::variants() {
                let name = format!("knn-1-{}", algorithm.name());
                let id = BenchmarkId::new(name, k);
                group.bench_with_input(id, &k, |b, &k| {
                    b.iter_with_large_drop(|| cakes.batch_knn_search(&queries, k, algorithm));
                });
            }
        }
        drop(cakes);

        for num_shards in (1..3).map(|i| 10usize.pow(i)) {
            let data = VecDataset::new("sharded".to_string(), train_data.clone(), metric, false);
            let criteria = PartitionCriteria::default();
            let cakes = ShardedCakes::new(data, Some(seed), criteria, cardinality / num_shards, 10, 10);

            // Run benchmarks on multiple shards
            for k in (0..3).map(|i| 10usize.pow(i)) {
                let name = format!("knn-{}-{}", num_shards, cakes.fastest_algorithm.name());
                let id = BenchmarkId::new(name, k);
                group.bench_with_input(id, &k, |b, &k| {
                    b.iter_with_large_drop(|| cakes.batch_knn_search(&queries, k));
                });
            }

            drop(cakes);
        }

        group.finish();
    }
}

criterion_group!(benches, cakes);
criterion_main!(benches);
