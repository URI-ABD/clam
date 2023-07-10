use criterion::*;

use symagen::random_data;

use abd_clam::{
    cakes::{KnnAlgorithm, CAKES},
    cluster::PartitionCriteria,
    dataset::VecVec,
    utils::METRICS,
};

fn cakes(c: &mut Criterion) {
    for &(metric_name, metric) in METRICS {
        let mut group = c.benchmark_group(format!("knn-{metric_name}"));
        group.significance_level(0.025).sample_size(10);

        let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Linear);
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
        let cakes = CAKES::new(dataset, Some(seed), criteria);

        for k in [1, 10, 100] {
            let id = BenchmarkId::new("100k-10", k);
            group.bench_with_input(id, &k, |b, &k| {
                b.iter_with_large_drop(|| cakes.batch_knn_search(&queries, k, KnnAlgorithm::RepeatedRnn));
            });
        }

        group.finish();
    }
}

criterion_group!(benches, cakes);
criterion_main!(benches);
