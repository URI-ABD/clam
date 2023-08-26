use core::cmp::Ordering;

use criterion::*;
use distances::Number;
use rayon::prelude::*;

use abd_clam::{knn, rnn, Cakes, Dataset, PartitionCriteria, ShardedCakes, VecDataset};

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
        if data_name != "deep-image" {
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

        let dataset = VecDataset::new("full".to_string(), train_data.clone(), metric, false);

        let min_cardinality = dataset.cardinality().as_f64().log2().ceil() as usize;
        let criteria = PartitionCriteria::new(true).with_min_cardinality(min_cardinality);

        let cakes = Cakes::new(dataset, Some(seed), criteria);

        // Run benchmarks on a single shard
        let ks_radii = (0..7)
            .map(|v| 2usize.pow(v))
            .map(|k| {
                let radii = queries
                    .par_iter()
                    .map(|query| {
                        cakes
                            .knn_search(query, k, knn::Algorithm::RepeatedRnn)
                            .into_iter()
                            .map(|(_, d)| d)
                            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                (k, radii)
            })
            .collect::<Vec<_>>();

        for (k, radii) in ks_radii.iter() {
            for &algorithm in rnn::Algorithm::variants() {
                if matches!(algorithm, rnn::Algorithm::Linear) {
                    continue;
                }

                let name = format!("rnn-1-{}", algorithm.name());
                let id = BenchmarkId::new(name, k);
                group.bench_with_input(id, &k, |b, _| {
                    b.iter_with_large_drop(|| {
                        queries
                            .par_iter()
                            .zip(radii.par_iter())
                            .map(|(&query, &radius)| cakes.rnn_search(query, radius, algorithm))
                            .collect::<Vec<_>>()
                    });
                });
            }

            for &algorithm in knn::Algorithm::variants() {
                // if !matches!(algorithm, knn::Algorithm::SieveSepCenter) {
                //     continue;
                // }

                let name = format!("knn-1-{}", algorithm.name());
                let id = BenchmarkId::new(name, k);
                group.bench_with_input(id, &k, |b, &&k| {
                    b.iter_with_large_drop(|| cakes.batch_knn_search(&queries, k, algorithm));
                });
            }
        }
        drop(cakes);

        // Run benchmarks on multiple shards
        for num_shards in (1..3).map(|v| 10usize.pow(v)) {
            let shards = train_data
                .chunks(cardinality / num_shards)
                .enumerate()
                .map(|(i, data)| {
                    let name = format!("shard-{}", i);
                    VecDataset::new(name, data.to_vec(), metric, false)
                })
                .map(|s| {
                    let min_cardinality = s.cardinality().as_f64().log2().ceil() as usize;
                    let criteria = PartitionCriteria::new(true).with_min_cardinality(min_cardinality);
                    (s, criteria)
                })
                .rev()
                .collect::<Vec<_>>();
            let ns = shards.len();

            let sharded_cakes = ShardedCakes::new(shards, Some(seed));

            for (k, radii) in ks_radii.iter() {
                for &algorithm in rnn::Algorithm::variants() {
                    let name = format!("rnn-{ns}-{}", algorithm.name());
                    let id = BenchmarkId::new(name, k);
                    group.bench_with_input(id, &k, |b, _| {
                        b.iter_with_large_drop(|| {
                            queries
                                .par_iter()
                                .zip(radii.par_iter())
                                .map(|(&query, &radius)| sharded_cakes.rnn_search(query, radius, algorithm))
                                .collect::<Vec<_>>()
                        });
                    });
                }

                for &algorithm in knn::Algorithm::variants() {
                    let name = format!("knn-{ns}-{}", algorithm.name());
                    let id = BenchmarkId::new(name, k);
                    group.bench_with_input(id, &k, |b, &&k| {
                        b.iter_with_large_drop(|| sharded_cakes.batch_knn_search(&queries, k, algorithm));
                    });
                }
            }
        }

        group.finish();
    }
}

criterion_group!(benches, cakes);
criterion_main!(benches);
