//! Benchmarking utilities for comparing the performance of different algorithms
//! on permuted datasets.

use abd_clam::{
    cakes::{
        cluster::{ParSearchable, Searchable},
        Algorithm, Decodable, Encodable, OffBall, ParCompressible, ParDecompressible, SquishyBall,
    },
    linear_search::ParLinearSearch,
    Ball,
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
/// - `Co`: The type of the original dataset.
pub fn compare_permuted<I, U, Co, Dec>(
    c: &mut Criterion,
    data_name: &str,
    metric_name: &str,
    data: &Co,
    root: &Ball<I, U, Co>,
    perm_data: &Co,
    perm_root: &OffBall<I, U, Co, Ball<I, U, Co>>,
    dec_data: &Dec,
    dec_root: &SquishyBall<I, U, Co, Dec, Ball<I, U, Co>>,
    queries: &[I],
    radii: &[U],
    ks: &[usize],
    par_only: bool,
    squishy: bool,
) where
    I: Encodable + Decodable + Send + Sync,
    U: Number,
    Co: ParCompressible<I, U> + ParLinearSearch<I, U>,
    Dec: ParDecompressible<I, U>,
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
            group.bench_with_input(BenchmarkId::new("Linear", radius), &radius, |b, _| {
                b.iter_with_large_drop(|| alg.batch_linear_search(data, queries));
            });
            group.bench_with_input(BenchmarkId::new("Ball", radius), &radius, |b, _| {
                b.iter_with_large_drop(|| root.batch_search(data, queries, alg));
            });
            group.bench_with_input(BenchmarkId::new("OffBall", radius), &radius, |b, _| {
                b.iter_with_large_drop(|| perm_root.batch_search(perm_data, queries, alg));
            });
            if squishy {
                group.bench_with_input(BenchmarkId::new("SquishyBall", radius), &radius, |b, _| {
                    b.iter_with_large_drop(|| dec_root.batch_search(dec_data, queries, alg));
                });
            }
        }

        group.bench_with_input(BenchmarkId::new("ParLinear", radius), &radius, |b, _| {
            b.iter_with_large_drop(|| alg.par_batch_linear_search(data, queries));
        });
        group.bench_with_input(BenchmarkId::new("ParBall", radius), &radius, |b, _| {
            b.iter_with_large_drop(|| root.par_batch_search(data, queries, alg));
        });
        group.bench_with_input(BenchmarkId::new("ParOffBall", radius), &radius, |b, _| {
            b.iter_with_large_drop(|| perm_root.par_batch_search(perm_data, queries, alg));
        });
        if squishy {
            group.bench_with_input(BenchmarkId::new("ParSquishyBall", radius), &radius, |b, _| {
                b.iter_with_large_drop(|| dec_root.par_batch_search(dec_data, queries, alg));
            });
        }
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
                group.bench_with_input(BenchmarkId::new("Linear", k), &k, |b, _| {
                    b.iter_with_large_drop(|| alg.batch_linear_search(data, queries));
                });
                group.bench_with_input(BenchmarkId::new("Ball", k), &k, |b, _| {
                    b.iter_with_large_drop(|| root.batch_search(data, queries, alg));
                });
                group.bench_with_input(BenchmarkId::new("OffBall", k), &k, |b, _| {
                    b.iter_with_large_drop(|| perm_root.batch_search(perm_data, queries, alg));
                });
                if squishy {
                    group.bench_with_input(BenchmarkId::new("SquishyBall", k), &k, |b, _| {
                        b.iter_with_large_drop(|| dec_root.batch_search(dec_data, queries, alg));
                    });
                }
            }

            group.bench_with_input(BenchmarkId::new("ParLinear", k), &k, |b, _| {
                b.iter_with_large_drop(|| alg.par_batch_linear_search(data, queries));
            });
            group.bench_with_input(BenchmarkId::new("ParBall", k), &k, |b, _| {
                b.iter_with_large_drop(|| root.par_batch_search(data, queries, alg));
            });
            group.bench_with_input(BenchmarkId::new("ParOffBall", k), &k, |b, _| {
                b.iter_with_large_drop(|| perm_root.par_batch_search(perm_data, queries, alg));
            });
            if squishy {
                group.bench_with_input(BenchmarkId::new("ParSquishyBall", k), &k, |b, _| {
                    b.iter_with_large_drop(|| dec_root.par_batch_search(dec_data, queries, alg));
                });
            }
        }
        group.finish();
    }
}
