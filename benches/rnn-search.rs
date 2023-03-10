use criterion::*;

use clam::core::cluster_criteria::PartitionCriteria;
use clam::core::dataset::VecVec;
use clam::search::cakes::CAKES;

pub mod utils;

use utils::distances::METRICS;

fn cakes(c: &mut Criterion) {
    for (metric_name, metric) in METRICS {
        let mut group = c.benchmark_group(format!("rnn-{metric_name}"));
        group.significance_level(0.025).sample_size(10);

        let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
        group.plot_config(plot_config);

        group.sampling_mode(SamplingMode::Flat);

        let num_queries = 10_000;
        group.throughput(Throughput::Elements(num_queries as u64));

        let (data, queries, name) = utils::make_data(1_000, 100, num_queries);
        let data = VecVec::new(data, metric, name, false);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);
        let cakes = CAKES::new(&data, Some(42)).build(&criteria);

        let queries = (0..num_queries).map(|i| &queries[i]).collect::<Vec<_>>();

        for n in (0..=100).step_by(10) {
            let radius = (n as f32) / if metric_name == "cosine" { 10_000. } else { 1_000. };

            let id = BenchmarkId::new("1M-100", radius);
            group.bench_with_input(id, &n, |b, _| {
                b.iter_with_large_drop(|| cakes.batch_rnn_search(&queries, radius));
            });

            let id = BenchmarkId::new("par-1M-100", radius);
            group.bench_with_input(id, &n, |b, _| {
                b.iter_with_large_drop(|| cakes.par_batch_rnn_search(&queries, radius));
            });
        }
        group.finish();
    }
}

criterion_group!(benches, cakes);
criterion_main!(benches);
