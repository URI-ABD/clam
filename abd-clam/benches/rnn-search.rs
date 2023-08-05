use criterion::*;

use symagen::random_data;

use abd_clam::{rnn, Cakes, PartitionCriteria, VecDataset, COMMON_METRICS_F32};

fn cakes(c: &mut Criterion) {
    let seed = 42;
    let (cardinality, dimensionality) = (100_000, 100);
    let (min_val, max_val) = (-1., 1.);

    let data = random_data::random_f32(cardinality, dimensionality, min_val, max_val, seed);
    let data = data.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let num_queries = 1_000;
    let queries = random_data::random_f32(num_queries, dimensionality, min_val, max_val, seed + 1);
    let queries = queries.iter().map(Vec::as_slice).collect::<Vec<_>>();

    for &(metric_name, metric) in &COMMON_METRICS_F32[..1] {
        let mut group = c.benchmark_group(format!("rnn-{metric_name}"));
        group
            .sample_size(10)
            .sampling_mode(SamplingMode::Flat)
            .throughput(Throughput::Elements(num_queries as u64))
            .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

        let dataset = VecDataset::new("rnn".to_string(), data.clone(), metric, false);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);
        let cakes = Cakes::new(dataset, Some(seed), criteria);

        let mut radius = 0.;
        for n in (0..=100).step_by(25) {
            radius = (n as f32) / if metric_name == "cosine" { 10_000. } else { 1_000. };
            let id = BenchmarkId::new("Clustered", radius);
            group.bench_with_input(id, &radius, |b, _| {
                b.iter_with_large_drop(|| cakes.batch_rnn_search(&queries, radius, rnn::Algorithm::Clustered));
            });
        }

        let id = BenchmarkId::new("Linear", radius);
        group.bench_with_input(id, &radius, |b, _| {
            b.iter_with_large_drop(|| cakes.batch_rnn_search(&queries, radius, rnn::Algorithm::Linear));
        });

        group.finish();
    }
}

criterion_group!(benches, cakes);
criterion_main!(benches);
