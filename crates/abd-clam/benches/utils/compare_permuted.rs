//! Benchmarking utilities for comparing the performance of different algorithms
//! on permuted datasets.

#![allow(unused_imports, unused_variables)]

use std::collections::HashMap;

use abd_clam::{
    cakes::{
        KnnBreadthFirst, KnnDepthFirst, KnnHinted, KnnLinear, KnnRepeatedRnn, ParSearchAlgorithm, ParSearchable,
        PermutedBall, RnnClustered, RnnLinear, SearchAlgorithm, Searchable,
    },
    dataset::ParDataset,
    metric::ParMetric,
    pancakes::{CodecData, Decodable, Encodable, ParCompressible, SquishyBall},
    Ball, Cluster, Dataset, FlatVec, Metric,
};
use criterion::*;
use distances::Number;
use measurement::WallTime;

/// Compare the performance of different algorithms on datasets using different
/// cluster types.
///
/// # Parameters
///
/// - `c`: The criterion context.
/// - `metric`: The metric used to measure distances.
/// - `ball_data`: The original dataset and its ball.
/// - `balanced_ball_data`: The original dataset and its balanced ball.
/// - `perm_ball_data`: The permuted dataset and its permuted ball.
/// - `perm_balanced_ball_data`: The permuted dataset and its permuted balanced
///   ball.
/// - `dec_ball_data`: The permuted dataset and its squishy ball, if any.
/// - `dec_balanced_ball_data`: The permuted dataset and its balanced squishy
///   ball, if any.
/// - `queries`: The queries to search for.
/// - `radii`: The radii to use for the RNN algorithm.
/// - `ks`: The numbers of neighbors to search for.
/// - `par_only`: Whether to only benchmark the parallel algorithms.
///
/// # Type Parameters
///
/// - `I`: The type of the items in the dataset.
/// - `U`: The type of the scalars used to measure distances.
/// - `Co`: The type of the compressible dataset.
/// - `M`: The type of the metric used to measure distances.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn compare_permuted<I, T, M, Me>(
    c: &mut Criterion,
    metric: &M,
    ball_data: (&Ball<T>, &FlatVec<I, (Me, HashMap<usize, T>)>),
    balanced_ball_data: (&Ball<T>, &FlatVec<I, (Me, HashMap<usize, T>)>),
    perm_ball_data: (&PermutedBall<T, Ball<T>>, &FlatVec<I, (Me, HashMap<usize, T>)>),
    perm_balanced_ball_data: (&PermutedBall<T, Ball<T>>, &FlatVec<I, (Me, HashMap<usize, T>)>),
    dec_ball_data: Option<(&SquishyBall<T, Ball<T>>, &CodecData<I, (Me, HashMap<usize, T>)>)>,
    dec_balanced_ball_data: Option<(&SquishyBall<T, Ball<T>>, &CodecData<I, (Me, HashMap<usize, T>)>)>,
    queries: &[I],
    radii: &[T],
    ks: &[usize],
    par_only: bool,
) where
    I: Encodable + Decodable + Send + Sync,
    T: Number + 'static,
    M: ParMetric<I, T>,
    Me: Send + Sync,
{
    let mut algs: Vec<(
        Box<dyn ParSearchAlgorithm<I, T, Ball<T>, M, FlatVec<I, (Me, HashMap<usize, T>)>>>,
        Box<dyn ParSearchAlgorithm<I, T, PermutedBall<T, Ball<T>>, M, FlatVec<I, (Me, HashMap<usize, T>)>>>,
        Option<Box<dyn ParSearchAlgorithm<I, T, SquishyBall<T, Ball<T>>, M, CodecData<I, (Me, HashMap<usize, T>)>>>>,
    )> = Vec::new();

    for (i, &radius) in radii.iter().enumerate() {
        if i == 0 {
            algs.push((
                Box::new(RnnLinear(radius)),
                Box::new(RnnLinear(radius)),
                Some(Box::new(RnnLinear(radius))),
            ));
        }
        algs.push((
            Box::new(RnnClustered(radius)),
            Box::new(RnnClustered(radius)),
            Some(Box::new(RnnClustered(radius))),
        ));
    }
    for (i, &k) in ks.iter().enumerate() {
        if i == 0 {
            algs.push((
                Box::new(KnnLinear(k)),
                Box::new(KnnLinear(k)),
                Some(Box::new(KnnLinear(k))),
            ));
        }
        algs.push((
            Box::new(KnnRepeatedRnn(k, T::ONE.double())),
            Box::new(KnnRepeatedRnn(k, T::ONE.double())),
            Some(Box::new(KnnRepeatedRnn(k, T::ONE.double()))),
        ));
        algs.push((
            Box::new(KnnBreadthFirst(k)),
            Box::new(KnnBreadthFirst(k)),
            Some(Box::new(KnnBreadthFirst(k))),
        ));
        algs.push((
            Box::new(KnnDepthFirst(k)),
            Box::new(KnnDepthFirst(k)),
            Some(Box::new(KnnDepthFirst(k))),
        ));
        algs.push((Box::new(KnnHinted(k)), Box::new(KnnHinted(k)), None));
    }

    for (alg_1, alg_2, alg_3) in &algs {
        let mut group = c.benchmark_group(format!("{}-{}", ball_data.1.name(), alg_1.name()));
        group
            .sample_size(10)
            .sampling_mode(SamplingMode::Flat)
            .throughput(Throughput::Elements(queries.len().as_u64()));

        if !par_only {
            bench_cakes(
                &mut group,
                alg_1,
                alg_2,
                alg_3.as_ref(),
                metric,
                queries,
                ball_data,
                balanced_ball_data,
                perm_ball_data,
                perm_balanced_ball_data,
                dec_ball_data,
                dec_balanced_ball_data,
            );
        }

        par_bench_cakes(
            &mut group,
            alg_1,
            alg_2,
            alg_3.as_ref(),
            metric,
            queries,
            ball_data,
            balanced_ball_data,
            perm_ball_data,
            perm_balanced_ball_data,
            dec_ball_data,
            dec_balanced_ball_data,
        );
        group.finish();
    }
}

fn bench_cakes<I, T, M, A1, A2, A3, Me>(
    group: &mut BenchmarkGroup<WallTime>,
    alg_1: &A1,
    alg_2: &A2,
    alg_3: Option<&A3>,
    metric: &M,
    queries: &[I],
    (ball, data): (&Ball<T>, &FlatVec<I, (Me, HashMap<usize, T>)>),
    (balanced_ball, balanced_data): (&Ball<T>, &FlatVec<I, (Me, HashMap<usize, T>)>),
    (perm_ball, perm_data): (&PermutedBall<T, Ball<T>>, &FlatVec<I, (Me, HashMap<usize, T>)>),
    (perm_balanced_ball, perm_balanced_data): (&PermutedBall<T, Ball<T>>, &FlatVec<I, (Me, HashMap<usize, T>)>),
    dec_ball_data: Option<(&SquishyBall<T, Ball<T>>, &CodecData<I, (Me, HashMap<usize, T>)>)>,
    dec_balanced_ball_data: Option<(&SquishyBall<T, Ball<T>>, &CodecData<I, (Me, HashMap<usize, T>)>)>,
) where
    I: Encodable + Decodable,
    T: Number,
    M: Metric<I, T>,
    A1: SearchAlgorithm<I, T, Ball<T>, M, FlatVec<I, (Me, HashMap<usize, T>)>>,
    A2: SearchAlgorithm<I, T, PermutedBall<T, Ball<T>>, M, FlatVec<I, (Me, HashMap<usize, T>)>>,
    A3: SearchAlgorithm<I, T, SquishyBall<T, Ball<T>>, M, CodecData<I, (Me, HashMap<usize, T>)>>,
{
    let parameter = if let Some(k) = alg_1.k() {
        k
    } else if let Some(radius) = alg_1.radius() {
        (ball.radius() / radius).as_usize()
    } else {
        0
    };

    group.bench_with_input(BenchmarkId::new("Ball", parameter), &0, |b, _| {
        b.iter_with_large_drop(|| alg_1.batch_search(data, metric, ball, queries));
    });

    group.bench_with_input(BenchmarkId::new("BalancedBall", parameter), &0, |b, _| {
        b.iter_with_large_drop(|| alg_1.batch_search(balanced_data, metric, balanced_ball, queries));
    });

    group.bench_with_input(BenchmarkId::new("PermBall", parameter), &0, |b, _| {
        b.iter_with_large_drop(|| alg_2.batch_search(perm_data, metric, perm_ball, queries));
    });

    group.bench_with_input(BenchmarkId::new("PermBalancedBall", parameter), &0, |b, _| {
        b.iter_with_large_drop(|| alg_2.batch_search(perm_balanced_data, metric, perm_balanced_ball, queries));
    });

    if let Some((dec_root, dec_data)) = dec_ball_data {
        group.bench_with_input(BenchmarkId::new("SquishyBall", parameter), &0, |b, _| {
            b.iter_with_large_drop(|| alg_3.map(|alg_3| alg_3.batch_search(dec_data, metric, dec_root, queries)));
        });
    }

    if let Some((dec_root, dec_data)) = dec_balanced_ball_data {
        group.bench_with_input(BenchmarkId::new("BalancedSquishyBall", parameter), &0, |b, _| {
            b.iter_with_large_drop(|| alg_3.map(|alg_3| alg_3.batch_search(dec_data, metric, dec_root, queries)));
        });
    }
}

fn par_bench_cakes<I, T, M, A1, A2, A3, Me>(
    group: &mut BenchmarkGroup<WallTime>,
    alg_1: &A1,
    alg_2: &A2,
    alg_3: Option<&A3>,
    metric: &M,
    queries: &[I],
    (ball, data): (&Ball<T>, &FlatVec<I, (Me, HashMap<usize, T>)>),
    (balanced_ball, balanced_data): (&Ball<T>, &FlatVec<I, (Me, HashMap<usize, T>)>),
    (perm_ball, perm_data): (&PermutedBall<T, Ball<T>>, &FlatVec<I, (Me, HashMap<usize, T>)>),
    (perm_balanced_ball, perm_balanced_data): (&PermutedBall<T, Ball<T>>, &FlatVec<I, (Me, HashMap<usize, T>)>),
    dec_ball_data: Option<(&SquishyBall<T, Ball<T>>, &CodecData<I, (Me, HashMap<usize, T>)>)>,
    dec_balanced_ball_data: Option<(&SquishyBall<T, Ball<T>>, &CodecData<I, (Me, HashMap<usize, T>)>)>,
) where
    I: Encodable + Decodable + Send + Sync,
    T: Number,
    M: ParMetric<I, T>,
    A1: ParSearchAlgorithm<I, T, Ball<T>, M, FlatVec<I, (Me, HashMap<usize, T>)>>,
    A2: ParSearchAlgorithm<I, T, PermutedBall<T, Ball<T>>, M, FlatVec<I, (Me, HashMap<usize, T>)>>,
    A3: ParSearchAlgorithm<I, T, SquishyBall<T, Ball<T>>, M, CodecData<I, (Me, HashMap<usize, T>)>>,
    Me: Send + Sync,
{
    let parameter = if let Some(k) = alg_1.k() {
        k
    } else if let Some(radius) = alg_1.radius() {
        (ball.radius() / radius).as_usize()
    } else {
        0
    };

    group.bench_with_input(BenchmarkId::new("ParBall", parameter), &0, |b, _| {
        b.iter_with_large_drop(|| alg_1.par_batch_search(data, metric, ball, queries));
    });

    group.bench_with_input(BenchmarkId::new("ParBalancedBall", parameter), &0, |b, _| {
        b.iter_with_large_drop(|| alg_1.par_batch_search(balanced_data, metric, balanced_ball, queries));
    });

    group.bench_with_input(BenchmarkId::new("ParPermBall", parameter), &0, |b, _| {
        b.iter_with_large_drop(|| alg_2.par_batch_search(perm_data, metric, perm_ball, queries));
    });

    group.bench_with_input(BenchmarkId::new("ParPermBalancedBall", parameter), &0, |b, _| {
        b.iter_with_large_drop(|| alg_2.par_batch_search(perm_balanced_data, metric, perm_balanced_ball, queries));
    });

    if let Some((dec_root, dec_data)) = dec_ball_data {
        group.bench_with_input(BenchmarkId::new("ParSquishyBall", parameter), &0, |b, _| {
            b.iter_with_large_drop(|| alg_3.map(|alg_3| alg_3.par_batch_search(dec_data, metric, dec_root, queries)));
        });
    }

    if let Some((dec_root, dec_data)) = dec_balanced_ball_data {
        group.bench_with_input(BenchmarkId::new("ParBalancedSquishyBall", parameter), &0, |b, _| {
            b.iter_with_large_drop(|| alg_3.map(|alg_3| alg_3.par_batch_search(dec_data, metric, dec_root, queries)));
        });
    }
}
