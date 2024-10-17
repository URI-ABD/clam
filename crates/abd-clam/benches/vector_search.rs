//! Benchmark for vector search.

mod utils;

use abd_clam::{
    adapter::{Adapter, ParAdapter, ParBallAdapter},
    cakes::OffBall,
    partition::ParPartition,
    BalancedBall, Ball, Cluster, FlatVec, Metric, Permutable,
};
use criterion::*;
use rand::prelude::*;

const METRICS: &[(&str, fn(&Vec<f32>, &Vec<f32>) -> f32)] = &[
    ("euclidean", |x: &Vec<_>, y: &Vec<_>| {
        distances::vectors::euclidean(x, y)
    }),
    ("cosine", |x: &Vec<_>, y: &Vec<_>| distances::vectors::cosine(x, y)),
];

fn vector_search(c: &mut Criterion) {
    let cardinality = 1_000_000;
    let dimensionality = 10;
    let max_val = 1.0;
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
    let radii = vec![0.001, 0.005, 0.01, 0.1];
    let ks = vec![1, 10, 100];
    for &(metric_name, distance_fn) in METRICS {
        let metric = Metric::new(distance_fn, true);
        let data = FlatVec::new(rows.clone(), metric).unwrap();

        let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
        let ball = Ball::par_new_tree(&data, &criteria, seed);
        let (off_ball, perm_data) = OffBall::par_from_ball_tree(ball.clone(), data.clone());

        let criteria = |c: &BalancedBall<_, _, _>| c.cardinality() > 1;
        let balanced_ball = BalancedBall::par_new_tree(&data, &criteria, seed);
        let (balanced_off_ball, balanced_perm_data) = {
            let balanced_off_ball = OffBall::par_adapt_tree(balanced_ball.clone(), None, &data);
            let mut balanced_perm_data = data.clone();
            let permutation = balanced_off_ball.source().indices().collect::<Vec<_>>();
            balanced_perm_data.permute(&permutation);
            (balanced_off_ball, balanced_perm_data)
        };

        utils::compare_permuted(
            c,
            "vector-search",
            metric_name,
            (&ball, &data),
            (&off_ball, &perm_data),
            None,
            (&balanced_ball, &data),
            (&balanced_off_ball, &balanced_perm_data),
            None,
            &queries,
            &radii,
            &ks,
            false,
        );
    }
}

criterion_group!(benches, vector_search);
criterion_main!(benches);
