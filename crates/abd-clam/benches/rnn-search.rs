use criterion::*;

use symagen::random_data;

use abd_clam::{rnn, Cakes, PartitionCriteria, VecDataset};

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
        let mut group = c.benchmark_group(format!("rnn-{metric_name}"));
        group
            .sample_size(100)
            .sampling_mode(SamplingMode::Flat)
            .throughput(Throughput::Elements(num_queries as u64))
            .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

        let dataset = VecDataset::<_, _, bool>::new("rnn".to_string(), data.clone(), metric, false, None);
        let criteria = PartitionCriteria::default();
        let cakes = Cakes::new(dataset, Some(seed), &criteria);

        let mut radius = 0.;
        for n in (0..=100).step_by(25) {
            radius = (n as f32) / if metric_name == "cosine" { 10_000. } else { 1_000. };

            for &variant in rnn::Algorithm::variants() {
                if matches!(variant, rnn::Algorithm::Linear) {
                    continue;
                }

                let id = BenchmarkId::new(variant.name(), radius);
                group.bench_with_input(id, &radius, |b, _| {
                    b.iter_with_large_drop(|| cakes.batch_rnn_search(&queries, radius, variant));
                });
            }
        }

        group.sample_size(10);
        let id = BenchmarkId::new("Linear", radius);
        group.bench_with_input(id, &radius, |b, _| {
            b.iter_with_large_drop(|| cakes.batch_rnn_search(&queries, radius, rnn::Algorithm::Linear));
        });

        group.finish();
    }
}

criterion_group!(benches, cakes);
criterion_main!(benches);
