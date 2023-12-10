use core::cmp::Ordering;

use criterion::*;
use rayon::prelude::*;
use symagen::random_data;

use abd_clam::{knn, rnn, Cakes, PartitionCriteria, VecDataset};

fn euclidean(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    distances::vectors::euclidean(x, y)
}

const METRICS: &[(&str, fn(&Vec<f32>, &Vec<f32>) -> f32)] = &[("euclidean", euclidean)];

fn cakes(c: &mut Criterion) {
    let seed = 42;
    let (cardinality, dimensionality) = (1_000_000, 10);
    let (min_val, max_val) = (-1., 1.);

    let data = random_data::random_tabular_seedable::<f32>(cardinality, dimensionality, min_val, max_val, seed);

    let num_queries = 100;
    let queries = random_data::random_tabular_seedable::<f32>(num_queries, dimensionality, min_val, max_val, seed + 1);
    let queries = queries.iter().collect::<Vec<_>>();

    for &(metric_name, metric) in METRICS {
        let mut group = c.benchmark_group(format!("knn-vs-rnn-{metric_name}"));
        group
            .sample_size(10)
            .sampling_mode(SamplingMode::Flat)
            .throughput(Throughput::Elements(num_queries as u64))
            .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

        let dataset = VecDataset::<_, _, bool>::new("knn".to_string(), data.clone(), metric, false, None);
        let criteria = PartitionCriteria::default();
        let cakes = Cakes::new(dataset, Some(seed), &criteria);

        for k in (0..=8).map(|v| 2usize.pow(v)) {
            let radii = cakes
                .batch_knn_search(&queries, k, knn::Algorithm::Linear)
                .into_iter()
                .map(|hits| {
                    hits.into_iter()
                        .map(|(_, d)| d)
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
                        .unwrap()
                })
                .collect::<Vec<_>>();

            for &variant in rnn::Algorithm::variants() {
                if matches!(variant, rnn::Algorithm::Linear) {
                    continue;
                }

                let id = BenchmarkId::new(format!("rnn-{}", variant.name()), k);
                group.bench_with_input(id, &k, |b, _| {
                    b.iter_with_large_drop(|| {
                        queries
                            .par_iter()
                            .zip(radii.par_iter())
                            .map(|(&query, &radius)| cakes.rnn_search(query, radius, variant))
                            .collect::<Vec<_>>()
                    });
                });
            }

            for &variant in knn::Algorithm::variants() {
                let id = BenchmarkId::new(variant.name(), k);
                group.bench_with_input(id, &k, |b, _| {
                    b.iter_with_large_drop(|| {
                        queries
                            .par_iter()
                            .zip(radii.par_iter())
                            .map(|(&query, _)| cakes.knn_search(query, k, variant))
                            .collect::<Vec<_>>()
                    });
                });
            }
        }

        group.finish();
    }
}

criterion_group!(benches, cakes);
criterion_main!(benches);
