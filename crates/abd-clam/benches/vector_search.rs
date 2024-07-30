use abd_clam::{
    cakes::{
        cluster::{ParSearchable, Searchable},
        Algorithm, OffsetBall,
    },
    dataset::ParDataset,
    partition::ParPartition,
    Ball, Cluster, FlatVec, Metric,
};
use criterion::*;
use distances::Number;
use rand::prelude::*;

const METRICS: &[(&str, fn(&Vec<f32>, &Vec<f32>) -> f32)] = &[
    ("euclidean", |x: &Vec<_>, y: &Vec<_>| {
        distances::vectors::euclidean(x, y)
    }),
    ("manhattan", |x: &Vec<_>, y: &Vec<_>| {
        distances::vectors::manhattan(x, y)
    }),
    ("cosine", |x: &Vec<_>, y: &Vec<_>| distances::vectors::cosine(x, y)),
];

fn vector_search(c: &mut Criterion) {
    let cardinality = 100_000;
    let dimensionality = 100;
    let max_val = 10.0;
    let min_val = -max_val;
    let seed = 42;
    let rows = symagen::random_data::random_tabular_seedable(cardinality, dimensionality, min_val, max_val, seed);

    let num_queries = 30;
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
    let radii = vec![0.01, 0.05, 0.1, 0.5];
    let ks = vec![1, 10, 100];
    for &(metric_name, distance_fn) in METRICS {
        let metric = Metric::new(distance_fn, true);
        let data = FlatVec::new(rows.clone(), metric).unwrap();

        let criteria = |c: &Ball<_>| c.cardinality() > 1;
        let root = Ball::par_new_tree(&data, &criteria, seed);

        let mut perm_data = data.clone();
        let perm_root = OffsetBall::par_from_ball_tree(root.clone(), &mut perm_data);

        compare_permuted(
            c,
            metric_name,
            &data,
            &root,
            &perm_data,
            &perm_root,
            &queries,
            &radii,
            &ks,
        );
    }
}

fn compare_permuted<I, U, D, Dp>(
    c: &mut Criterion,
    metric_name: &str,
    data: &D,
    root: &Ball<U>,
    perm_data: &Dp,
    perm_root: &OffsetBall<U>,
    queries: &[I],
    radii: &[U],
    ks: &[usize],
) where
    I: Send + Sync,
    U: Number,
    D: ParDataset<I, U>,
    Dp: ParDataset<I, U>,
{
    let algs = vec![
        Algorithm::KnnRepeatedRnn(ks[0], U::ONE.double()),
        Algorithm::KnnBreadthFirst(ks[0]),
        Algorithm::KnnDepthFirst(ks[0]),
    ];

    let mut group = c.benchmark_group(format!("vectors-RnnClustered-{}", metric_name));
    group
        .sample_size(10)
        .sampling_mode(SamplingMode::Flat)
        .throughput(Throughput::Elements(queries.len().as_u64()));

    for &radius in radii {
        let alg = Algorithm::RnnClustered(radius);

        group.bench_with_input(BenchmarkId::new("Ball", radius), &radius, |b, _| {
            b.iter_with_large_drop(|| root.batch_search(data, queries, alg));
        });
        group.bench_with_input(BenchmarkId::new("OffsetBall", radius), &radius, |b, _| {
            b.iter_with_large_drop(|| perm_root.batch_search(perm_data, queries, alg));
        });

        group.bench_with_input(BenchmarkId::new("ParBall", radius), &radius, |b, _| {
            b.iter_with_large_drop(|| root.par_batch_search(data, queries, alg));
        });
        group.bench_with_input(BenchmarkId::new("ParOffsetBall", radius), &radius, |b, _| {
            b.iter_with_large_drop(|| perm_root.par_batch_search(perm_data, queries, alg));
        });
    }
    group.finish();

    for alg in &algs {
        let mut group = c.benchmark_group(format!("vectors-{}-{}", alg.variant_name(), metric_name));
        group
            .sample_size(10)
            .sampling_mode(SamplingMode::Flat)
            .throughput(Throughput::Elements(queries.len().as_u64()));

        for &k in ks {
            let alg = alg.with_params(U::ZERO, k);

            group.bench_with_input(BenchmarkId::new("Ball", k), &k, |b, _| {
                b.iter_with_large_drop(|| root.batch_search(data, queries, alg));
            });
            group.bench_with_input(BenchmarkId::new("OffsetBall", k), &k, |b, _| {
                b.iter_with_large_drop(|| perm_root.batch_search(perm_data, queries, alg));
            });
            group.bench_with_input(BenchmarkId::new("ParBall", k), &k, |b, _| {
                b.iter_with_large_drop(|| root.par_batch_search(data, queries, alg));
            });
            group.bench_with_input(BenchmarkId::new("ParOffsetBall", k), &k, |b, _| {
                b.iter_with_large_drop(|| perm_root.par_batch_search(perm_data, queries, alg));
            });
        }
        group.finish();
    }
}

criterion_group!(benches, vector_search);
criterion_main!(benches);
