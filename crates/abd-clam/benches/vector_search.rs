//! Benchmark for vector search.

use std::collections::HashMap;

use abd_clam::{
    cakes::{HintedDataset, PermutedBall},
    cluster::{adapter::ParBallAdapter, BalancedBall, ParPartition},
    dataset::AssociatesMetadataMut,
    metric::{Euclidean, ParMetric},
    Ball, Cluster, Dataset, FlatVec, Metric,
};
use criterion::*;
use rand::prelude::*;
use utils::Row;

mod utils;

/// The Euclidean metric using SIMD instructions.
pub struct EuclideanSimd;

impl<I: AsRef<[f32]>> Metric<I, f32> for EuclideanSimd {
    fn distance(&self, a: &I, b: &I) -> f32 {
        distances::simd::euclidean_f32(a.as_ref(), b.as_ref())
    }

    fn name(&self) -> &str {
        "euclidean-simd"
    }

    fn has_identity(&self) -> bool {
        true
    }

    fn has_non_negativity(&self) -> bool {
        true
    }

    fn has_symmetry(&self) -> bool {
        true
    }

    fn obeys_triangle_inequality(&self) -> bool {
        true
    }

    fn is_expensive(&self) -> bool {
        false
    }
}

impl<I: AsRef<[f32]> + Send + Sync> ParMetric<I, f32> for EuclideanSimd {}

fn run_search<M: ParMetric<Row<f32>, f32>>(
    c: &mut Criterion,
    data: FlatVec<Row<f32>, usize>,
    metric: &M,
    queries: &[Row<f32>],
    radii: &[f32],
    ks: &[usize],
    seed: Option<u64>,
) {
    let criteria = |c: &Ball<_>| c.cardinality() > 1;
    let ball = Ball::par_new_tree(&data, metric, &criteria, seed);
    let radii = radii.iter().map(|&r| r * ball.radius()).collect::<Vec<_>>();

    let criteria = |c: &BalancedBall<_>| c.cardinality() > 1;
    let balanced_ball = BalancedBall::par_new_tree(&data, metric, &criteria, seed).into_ball();

    let (_, max_radius) = abd_clam::utils::arg_max(&radii).unwrap();
    let (_, max_k) = abd_clam::utils::arg_max(ks).unwrap();
    let data = data
        .transform_metadata(|&i| (i, HashMap::new()))
        .with_hints_from_tree(&ball, metric)
        .with_hints_from(metric, &balanced_ball, max_radius, max_k);

    let (perm_ball, perm_data) = PermutedBall::par_from_ball_tree(ball.clone(), data.clone(), metric);
    let (perm_balanced_ball, perm_balanced_data) =
        PermutedBall::par_from_ball_tree(balanced_ball.clone(), data.clone(), metric);

    utils::compare_permuted(
        c,
        metric,
        (&ball, &data),
        (&balanced_ball, &perm_balanced_data),
        (&perm_ball, &perm_data),
        (&perm_balanced_ball, &perm_balanced_data),
        None,
        None,
        queries,
        &radii,
        ks,
        true,
    );
}

fn vector_search(c: &mut Criterion) {
    let cardinality = 1_000_000;
    let dimensionality = 10;
    let max_val = 1.0;
    let min_val = -max_val;
    let seed = 42;
    let rows = symagen::random_data::random_tabular_seedable(cardinality, dimensionality, min_val, max_val, seed)
        .into_iter()
        .map(Row::from)
        .collect::<Vec<_>>();

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

    let data = FlatVec::new(rows)
        .unwrap_or_else(|e| unreachable!("{e}"))
        .with_name("vector-search");
    let seed = Some(seed);
    let radii = vec![0.001, 0.01];
    let ks = vec![1, 10, 100];

    run_search(c, data, &Euclidean, &queries, &radii, &ks, seed);
    // run_search(c, &data, &EuclideanSimd, &queries, &radii, &ks, seed);
}

criterion_group!(benches, vector_search);
criterion_main!(benches);
