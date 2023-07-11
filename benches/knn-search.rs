use criterion::*;

use symagen::random_data;

use abd_clam::{
    cakes::{KnnAlgorithm, CAKES},
    cluster::PartitionCriteria,
    dataset::VecDataset,
    COMMON_METRICS_F32,
};

fn cakes(c: &mut Criterion) {
    let seed = 42;
    let (cardinality, dimensionality) = (10_000, 100);
    let (min_val, max_val) = (-1., 1.);
    let data = random_data::random_f32(cardinality, dimensionality, min_val, max_val, seed);
    let data = data.iter().map(|v| v.as_slice()).collect::<Vec<_>>();

    let num_queries = 100;
    let queries = random_data::random_f32(num_queries, dimensionality, min_val, max_val, seed + 1);
    let queries = queries.iter().map(|v| v.as_slice()).collect::<Vec<_>>();

    for &(metric_name, metric) in &COMMON_METRICS_F32[..1] {
        let mut group = c.benchmark_group(format!("knn-{metric_name}"));

        let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
        group.plot_config(plot_config);

        group.sampling_mode(SamplingMode::Flat);

        group.throughput(Throughput::Elements(num_queries as u64));

        let dataset = VecDataset::new(data.clone(), metric, "knn".to_string(), false);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);
        let cakes = CAKES::new(dataset, Some(seed), criteria);

        let ks = [100, 10, 1];
        for k in ks {
            let id = BenchmarkId::new("RepeatedRnn", k);
            group.bench_with_input(id, &k, |b, _| {
                b.iter_with_large_drop(|| cakes.batch_knn_search(&queries, k, KnnAlgorithm::RepeatedRnn));
            });
        }

        group.sample_size(10);
        let id = BenchmarkId::new("Linear", ks[0]);
        group.bench_with_input(id, &ks[0], |b, _| {
            b.iter_with_large_drop(|| cakes.batch_knn_search(&queries, ks[0], KnnAlgorithm::Linear));
        });

        group.finish();
    }
}

criterion_group!(benches, cakes);
criterion_main!(benches);
