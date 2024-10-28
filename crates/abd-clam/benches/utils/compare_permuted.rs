//! Benchmarking utilities for comparing the performance of different algorithms
//! on permuted datasets.

use abd_clam::{
    cakes::{
        cluster::{ParSearchable, Searchable},
        Algorithm, CodecData, Decodable, Encodable, OffBall, ParCompressible, SquishyBall,
    },
    BalancedBall, Ball,
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
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn compare_permuted<I, U, Co>(
    c: &mut Criterion,
    data_name: &str,
    metric_name: &str,
    ball_data: (&Ball<I, U, Co>, &Co),
    off_ball_data: (&OffBall<I, U, Co, Ball<I, U, Co>>, &Co),
    dec_ball_data: Option<(
        &SquishyBall<I, U, Co, CodecData<I, U, usize>, Ball<I, U, Co>>,
        &CodecData<I, U, usize>,
    )>,
    bal_ball_data: (&BalancedBall<I, U, Co>, &Co),
    bal_off_ball_data: (&OffBall<I, U, Co, BalancedBall<I, U, Co>>, &Co),
    bal_dec_ball_data: Option<(
        &SquishyBall<I, U, Co, CodecData<I, U, usize>, BalancedBall<I, U, Co>>,
        &CodecData<I, U, usize>,
    )>,
    queries: &[I],
    radii: &[U],
    ks: &[usize],
    par_only: bool,
) where
    I: Encodable + Decodable + Send + Sync,
    U: Number,
    Co: ParCompressible<I, U>,
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

    let (ball, data) = ball_data;
    let (off_ball, perm_data) = off_ball_data;

    let (bal_ball, bal_data) = bal_ball_data;
    let (bal_off_ball, bal_perm_data) = bal_off_ball_data;

    for &radius in radii {
        let alg = Algorithm::RnnClustered(radius);

        if !par_only {
            group.bench_with_input(BenchmarkId::new("Ball", radius), &radius, |b, _| {
                b.iter_with_large_drop(|| ball.batch_search(data, queries, alg));
            });
            group.bench_with_input(BenchmarkId::new("BalancedBall", radius), &radius, |b, _| {
                b.iter_with_large_drop(|| bal_ball.batch_search(bal_data, queries, alg));
            });

            group.bench_with_input(BenchmarkId::new("OffBall", radius), &radius, |b, _| {
                b.iter_with_large_drop(|| off_ball.batch_search(perm_data, queries, alg));
            });
            group.bench_with_input(BenchmarkId::new("BalancedOffBall", radius), &radius, |b, _| {
                b.iter_with_large_drop(|| bal_off_ball.batch_search(bal_perm_data, queries, alg));
            });

            if let Some((dec_root, dec_data)) = dec_ball_data {
                group.bench_with_input(BenchmarkId::new("SquishyBall", radius), &radius, |b, _| {
                    b.iter_with_large_drop(|| dec_root.batch_search(dec_data, queries, alg));
                });
            }
            if let Some((dec_root, dec_data)) = bal_dec_ball_data {
                group.bench_with_input(BenchmarkId::new("BalancedSquishyBall", radius), &radius, |b, _| {
                    b.iter_with_large_drop(|| dec_root.batch_search(dec_data, queries, alg));
                });
            }
        }

        group.bench_with_input(BenchmarkId::new("ParBall", radius), &radius, |b, _| {
            b.iter_with_large_drop(|| ball.par_batch_search(data, queries, alg));
        });
        group.bench_with_input(BenchmarkId::new("ParBalancedBall", radius), &radius, |b, _| {
            b.iter_with_large_drop(|| bal_ball.par_batch_search(bal_data, queries, alg));
        });

        group.bench_with_input(BenchmarkId::new("ParOffBall", radius), &radius, |b, _| {
            b.iter_with_large_drop(|| off_ball.par_batch_search(perm_data, queries, alg));
        });
        group.bench_with_input(BenchmarkId::new("ParBalancedOffBall", radius), &radius, |b, _| {
            b.iter_with_large_drop(|| bal_off_ball.par_batch_search(bal_perm_data, queries, alg));
        });

        if let Some((dec_root, dec_data)) = dec_ball_data {
            group.bench_with_input(BenchmarkId::new("ParSquishyBall", radius), &radius, |b, _| {
                b.iter_with_large_drop(|| dec_root.par_batch_search(dec_data, queries, alg));
            });
        }
        if let Some((dec_root, dec_data)) = bal_dec_ball_data {
            group.bench_with_input(BenchmarkId::new("ParBalancedSquishyBall", radius), &radius, |b, _| {
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
                group.bench_with_input(BenchmarkId::new("Ball", k), &k, |b, _| {
                    b.iter_with_large_drop(|| ball.batch_search(data, queries, alg));
                });
                group.bench_with_input(BenchmarkId::new("BalancedBall", k), &k, |b, _| {
                    b.iter_with_large_drop(|| bal_ball.batch_search(bal_data, queries, alg));
                });

                group.bench_with_input(BenchmarkId::new("OffBall", k), &k, |b, _| {
                    b.iter_with_large_drop(|| off_ball.batch_search(perm_data, queries, alg));
                });
                group.bench_with_input(BenchmarkId::new("BalancedOffBall", k), &k, |b, _| {
                    b.iter_with_large_drop(|| bal_off_ball.batch_search(bal_perm_data, queries, alg));
                });

                if let Some((dec_root, dec_data)) = dec_ball_data {
                    group.bench_with_input(BenchmarkId::new("SquishyBall", k), &k, |b, _| {
                        b.iter_with_large_drop(|| dec_root.batch_search(dec_data, queries, alg));
                    });
                }
                if let Some((dec_root, dec_data)) = bal_dec_ball_data {
                    group.bench_with_input(BenchmarkId::new("BalancedSquishyBall", k), &k, |b, _| {
                        b.iter_with_large_drop(|| dec_root.batch_search(dec_data, queries, alg));
                    });
                }
            }

            group.bench_with_input(BenchmarkId::new("ParBall", k), &k, |b, _| {
                b.iter_with_large_drop(|| ball.par_batch_search(data, queries, alg));
            });
            group.bench_with_input(BenchmarkId::new("ParBalancedBall", k), &k, |b, _| {
                b.iter_with_large_drop(|| bal_ball.par_batch_search(bal_data, queries, alg));
            });

            group.bench_with_input(BenchmarkId::new("ParOffBall", k), &k, |b, _| {
                b.iter_with_large_drop(|| off_ball.par_batch_search(perm_data, queries, alg));
            });
            group.bench_with_input(BenchmarkId::new("ParBalancedOffBall", k), &k, |b, _| {
                b.iter_with_large_drop(|| bal_off_ball.par_batch_search(bal_perm_data, queries, alg));
            });

            if let Some((dec_root, dec_data)) = dec_ball_data {
                group.bench_with_input(BenchmarkId::new("ParSquishyBall", k), &k, |b, _| {
                    b.iter_with_large_drop(|| dec_root.par_batch_search(dec_data, queries, alg));
                });
            }
            if let Some((dec_root, dec_data)) = bal_dec_ball_data {
                group.bench_with_input(BenchmarkId::new("ParBalancedSquishyBall", k), &k, |b, _| {
                    b.iter_with_large_drop(|| dec_root.par_batch_search(dec_data, queries, alg));
                });
            }
        }
        group.finish();
    }
}
