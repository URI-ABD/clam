//! Benchmarking utilities for comparing the performance of different algorithms
//! on permuted datasets.

use abd_clam::{
    cakes::{
        cluster::{ParSearchable, Searchable},
        Algorithm, OffsetBall,
    },
    dataset::ParDataset,
};
use criterion::*;
use distances::Number;

/// Compare the performance of different algorithms on permuted datasets.
///
/// # Parameters
///
/// - `c`: The criterion context.
/// - `metric_name`: The name of the metric used to measure distances.
/// - `data`: The original dataset.
/// - `root`: The root of tree on the original dataset.
/// - `perm_data`: The permuted dataset.
/// - `perm_root`: The root of the tree on the permuted dataset.
/// - `queries`: The queries to search for.
/// - `radii`: The radii to use for RNN algorithms.
/// - `ks`: The values of `k` to use for kNN algorithms.
///
/// # Type Parameters
///
/// - `I`: The type of the items in the dataset.
/// - `U`: The type of the scalars used to measure distances.
/// - `C`: The type of the cluster for the original dataset.
/// - `D`: The type of the original dataset.
/// - `Dp`: The type of the permuted dataset.
pub fn compare_permuted<I, U, C, D, Dp>(
    c: &mut Criterion,
    data_name: &str,
    metric_name: &str,
    data: &D,
    root: &C,
    perm_data: &Dp,
    perm_root: &OffsetBall<U, C>,
    queries: &[I],
    radii: &[U],
    ks: &[usize],
    par_only: bool,
) where
    I: Send + Sync,
    U: Number,
    C: ParSearchable<I, U, D>,
    D: ParDataset<I, U>,
    Dp: ParDataset<I, U>,
{
    let algs = vec![
        Algorithm::KnnRepeatedRnn(ks[0], U::ONE.double()),
        Algorithm::KnnBreadthFirst(ks[0]),
        Algorithm::KnnDepthFirst(ks[0]),
    ];

    let mut group = c.benchmark_group(format!("{}-{}-RnnClustered", data_name, metric_name));
    group
        .sample_size(10)
        .sampling_mode(SamplingMode::Flat)
        .throughput(Throughput::Elements(queries.len().as_u64()));

    for &radius in radii {
        let alg = Algorithm::RnnClustered(radius);

        if !par_only {
            group.bench_with_input(BenchmarkId::new("Ball", radius), &radius, |b, _| {
                b.iter_with_large_drop(|| root.batch_search(data, queries, alg));
            });
            group.bench_with_input(BenchmarkId::new("OffsetBall", radius), &radius, |b, _| {
                b.iter_with_large_drop(|| perm_root.batch_search(perm_data, queries, alg));
            });
        }

        group.bench_with_input(BenchmarkId::new("ParBall", radius), &radius, |b, _| {
            b.iter_with_large_drop(|| root.par_batch_search(data, queries, alg));
        });
        group.bench_with_input(BenchmarkId::new("ParOffsetBall", radius), &radius, |b, _| {
            b.iter_with_large_drop(|| perm_root.par_batch_search(perm_data, queries, alg));
        });
    }
    group.finish();

    for alg in &algs {
        let mut group = c.benchmark_group(format!("{}-{}-{}", data_name, metric_name, alg.variant_name()));
        group
            .sample_size(10)
            .sampling_mode(SamplingMode::Flat)
            .throughput(Throughput::Elements(queries.len().as_u64()));

        for &k in ks {
            let alg = alg.with_params(U::ZERO, k);

            if !par_only {
                group.bench_with_input(BenchmarkId::new("Ball", k), &k, |b, _| {
                    b.iter_with_large_drop(|| root.batch_search(data, queries, alg));
                });
                group.bench_with_input(BenchmarkId::new("OffsetBall", k), &k, |b, _| {
                    b.iter_with_large_drop(|| perm_root.batch_search(perm_data, queries, alg));
                });
            }

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
