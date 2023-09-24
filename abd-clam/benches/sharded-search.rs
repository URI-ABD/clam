use criterion::*;

use abd_clam::{knn, Cakes, Dataset, PartitionCriteria, ShardedCakes, VecDataset};
use symagen::random_data;

fn sharded_cakes(c: &mut Criterion) {
    let (cardinality, dimensionality) = (2usize.pow(21), 10);
    let num_queries = 100;
    let (min_val, max_val) = (-1., 1.);
    let seed = 42;

    let metric = distances::simd::euclidean_f32;
    let metric_name = "euclidean";

    let ks = (0..3).map(|i| 10usize.pow(i)).collect::<Vec<_>>();
    let shard_numbers = (1..4).map(|i| 2usize.pow(i)).collect::<Vec<_>>();

    let mut group = c.benchmark_group(format!("car-{cardinality}-dim-{dimensionality}-{metric_name}"));
    group
        .sample_size(10)
        .sampling_mode(SamplingMode::Flat)
        .throughput(Throughput::Elements(num_queries as u64))
        .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let train_data = random_data::random_f32(cardinality, dimensionality, min_val, max_val, seed);
    let train_data = train_data.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let queries = random_data::random_f32(num_queries, dimensionality, min_val, max_val, seed + 1);
    let queries = queries.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let data = VecDataset::new("full".to_string(), train_data.clone(), metric, false);
    let criteria = PartitionCriteria::default();
    let cakes = Cakes::new(data, Some(seed), criteria).auto_tune(10, 10);

    // Run benchmarks on a single shard
    for &k in ks.iter() {
        for &algorithm in knn::Algorithm::variants() {
            let name = format!("sharded-1-{}", algorithm.name());
            let id = BenchmarkId::new(name, k);
            group.bench_with_input(id, &k, |b, &k| {
                b.iter_with_large_drop(|| cakes.batch_knn_search(&queries, k, algorithm));
            });
        }
    }
    drop(cakes);

    for &num_shards in shard_numbers.iter() {
        let max_cardinality = cardinality / num_shards;
        let shards = VecDataset::new("sharded".to_string(), train_data.clone(), metric, false)
            .make_shards(max_cardinality)
            .into_iter()
            .map(|data| Cakes::new(data, Some(seed), PartitionCriteria::default()))
            .collect();
        let cakes = ShardedCakes::new(shards).auto_tune(10, 10);

        // Run benchmarks on multiple shards
        for &k in ks.iter() {
            let name = format!("sharded-{}-{}", num_shards, cakes.best_knn_algorithm().name());
            let id = BenchmarkId::new(name, k);
            group.bench_with_input(id, &k, |b, &k| {
                b.iter_with_large_drop(|| cakes.batch_knn_search(&queries, k));
            });
        }

        drop(cakes);
    }

    group.finish();
}

criterion_group!(benches, sharded_cakes);
criterion_main!(benches);
