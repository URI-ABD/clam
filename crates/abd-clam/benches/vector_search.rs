use abd_clam::{
    cakes::{self, Algorithm, OffsetBall},
    partition::ParPartition,
    Ball, Cluster, FlatVec, Metric,
};
use criterion::*;
use distances::Number;
use rand::prelude::*;

const METRICS: &[(&str, fn(&Vec<f32>, &Vec<f32>) -> f32)] = &[
    ("euclidean", |x: &Vec<f32>, y: &Vec<f32>| {
        distances::vectors::euclidean(x, y)
    }),
    ("euclidean_sq", |x: &Vec<f32>, y: &Vec<f32>| {
        distances::vectors::euclidean_sq(x, y)
    }),
    ("manhattan", |x: &Vec<f32>, y: &Vec<f32>| {
        distances::vectors::manhattan(x, y)
    }),
    ("cosine", |x: &Vec<f32>, y: &Vec<f32>| distances::vectors::cosine(x, y)),
];

fn vector_search(c: &mut Criterion) {
    let cardinality = 100_000;
    let dimensionality = 100;
    let max_val = 10.0;
    let min_val = -max_val;
    let seed = 42;
    let rows = symagen::random_data::random_tabular_seedable(cardinality, dimensionality, min_val, max_val, seed);

    let num_queries = 20;
    let queries = {
        let mut indices = (0..rows.len()).collect::<Vec<_>>();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
        indices
            .into_iter()
            .take(num_queries)
            .map(|i| rows[i].clone())
            .collect::<Vec<_>>()
    };

    let seed = Some(seed);
    let radii = vec![0.05, 0.1, 0.5, 1.0];
    let ks = vec![1, 10, 100];
    for &(metric_name, distance_fn) in METRICS {
        let metric = Metric::new(distance_fn, true);
        let data = FlatVec::new(rows.clone(), metric).unwrap();

        let criteria = |c: &Ball<f32>| c.cardinality() > 1;
        let root = Ball::par_new_tree(&data, &criteria, seed);
        bench_on_root(c, false, metric_name, &root, &data, &queries, &radii, &ks);

        let mut data = data;
        let root = OffsetBall::par_from_ball_tree(root, &mut data);
        bench_on_root(c, true, metric_name, &root, &data, &queries, &radii, &ks);
    }
}

fn bench_on_root<C>(
    c: &mut Criterion,
    permuted: bool,
    metric_name: &str,
    root: &C,
    data: &FlatVec<Vec<f32>, f32, usize>,
    queries: &[Vec<f32>],
    radii: &[f32],
    ks: &[usize],
) where
    C: Cluster<f32> + cakes::cluster::ParSearchable<Vec<f32>, f32, FlatVec<Vec<f32>, f32, usize>>,
{
    let permuted = if permuted { "-permuted" } else { "" };

    let mut group = c.benchmark_group(format!("vector-search-{}{}", metric_name, permuted));
    group
        .sample_size(10)
        .sampling_mode(SamplingMode::Flat)
        .throughput(Throughput::Elements(queries.len().as_u64()))
        .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &radius in radii {
        let id = BenchmarkId::new("RnnClustered", radius);
        group.bench_with_input(id, &radius, |b, &radius| {
            b.iter_with_large_drop(|| root.batch_search(&data, &queries, Algorithm::RnnClustered(radius)));
        });
    }

    for &k in ks {
        let id = BenchmarkId::new("KnnRepeatedRnn", k);
        group.bench_with_input(id, &k, |b, &k| {
            b.iter_with_large_drop(|| root.batch_search(&data, &queries, Algorithm::KnnRepeatedRnn(k, 2.0)));
        });

        let id = BenchmarkId::new("KnnBreadthFirst", k);
        group.bench_with_input(id, &k, |b, &k| {
            b.iter_with_large_drop(|| root.batch_search(&data, &queries, Algorithm::KnnBreadthFirst(k)));
        });

        let id = BenchmarkId::new("KnnDepthFirst", k);
        group.bench_with_input(id, &k, |b, &k| {
            b.iter_with_large_drop(|| root.batch_search(&data, &queries, Algorithm::KnnDepthFirst(k)));
        });
    }

    for &radius in radii {
        let id = BenchmarkId::new("ParRnnClustered", radius);
        group.bench_with_input(id, &radius, |b, &radius| {
            b.iter_with_large_drop(|| root.par_batch_par_search(&data, queries, Algorithm::RnnClustered(radius)));
        });
    }

    for &k in ks {
        let id = BenchmarkId::new("ParKnnRepeatedRnn", k);
        group.bench_with_input(id, &k, |b, &k| {
            b.iter_with_large_drop(|| root.par_batch_par_search(&data, &queries, Algorithm::KnnRepeatedRnn(k, 2.0)));
        });

        let id = BenchmarkId::new("ParKnnBreadthFirst", k);
        group.bench_with_input(id, &k, |b, &k| {
            b.iter_with_large_drop(|| root.par_batch_par_search(&data, &queries, Algorithm::KnnBreadthFirst(k)));
        });

        let id = BenchmarkId::new("ParKnnDepthFirst", k);
        group.bench_with_input(id, &k, |b, &k| {
            b.iter_with_large_drop(|| root.par_batch_par_search(&data, &queries, Algorithm::KnnDepthFirst(k)));
        });
    }
}

criterion_group!(benches, vector_search);
criterion_main!(benches);
