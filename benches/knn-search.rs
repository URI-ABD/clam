use criterion::*;

use clam::core::cluster_criteria::PartitionCriteria;
use clam::core::dataset::VecVec;
use clam::search::cakes::CAKES;

pub mod utils;

use utils::distances::METRICS;

fn cakes(c: &mut Criterion) {
    for (metric_name, metric) in METRICS {
        let mut group = c.benchmark_group(format!("knn-{metric_name}"));
        group.significance_level(0.025).sample_size(10);

        let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Linear);
        group.plot_config(plot_config);

        group.sampling_mode(SamplingMode::Flat);

        let num_queries = 10_000;
        group.throughput(Throughput::Elements(num_queries as u64));

        let (data, queries, name) = utils::make_data(1_000, 100, num_queries);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);

        let data = VecVec::new(data, metric, name, false);
        let cakes = CAKES::new(&data, Some(42)).build(&criteria);

        let queries = (0..num_queries).map(|i| &queries[i]).collect::<Vec<_>>();

        for k in [1, 10, 100] {
            let id = BenchmarkId::new("1M-100", k);
            group.bench_with_input(id, &k, |b, &k| {
                b.iter_with_large_drop(|| cakes.batch_knn_search(&queries, k));
            });

            let id = BenchmarkId::new("par-1M-100", k);
            group.bench_with_input(id, &k, |b, &k| {
                b.iter_with_large_drop(|| cakes.par_batch_knn_search(&queries, k));
            });
        }

        group.finish();
    }
}

criterion_group!(benches, cakes);
criterion_main!(benches);
