use criterion::*;

use symagen::random_data;

use abd_clam::{cakes::CAKES, cluster::PartitionCriteria, dataset::VecVec, utils::METRICS};

fn cakes(c: &mut Criterion) {
    for &(metric_name, metric) in METRICS {
        let mut group = c.benchmark_group(format!("rnn-{metric_name}"));
        group.significance_level(0.025).sample_size(10);

        let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
        group.plot_config(plot_config);

        group.sampling_mode(SamplingMode::Flat);

        let num_queries = 10_000;
        group.throughput(Throughput::Elements(num_queries as u64));

        let seed = 42;
        let (dimensionality, min_val, max_val) = (10, 0., 1.);
        let data = random_data::random_f32(100_000, dimensionality, min_val, max_val, seed);
        let queries = random_data::random_f32(num_queries, dimensionality, min_val, max_val, seed);
        let data = data.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
        let queries = queries.iter().map(|v| v.as_slice()).collect::<Vec<_>>();

        let dataset = VecVec::new(data, metric, "100k-10".to_string(), false);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);
        let cakes = CAKES::new(dataset, Some(seed)).build(criteria);

        for n in (0..=100).step_by(10) {
            let radius = (n as f32) / if metric_name == "cosine" { 10_000. } else { 1_000. };

            let id = BenchmarkId::new("100k-10", radius);
            group.bench_with_input(id, &n, |b, _| {
                b.iter_with_large_drop(|| cakes.batch_rnn_search(&queries, radius));
            });

            let id = BenchmarkId::new("par-100k-10", radius);
            group.bench_with_input(id, &n, |b, _| {
                b.iter_with_large_drop(|| cakes.par_batch_rnn_search(&queries, radius));
            });
        }

        group.finish();
    }
}

criterion_group!(benches, cakes);
criterion_main!(benches);
