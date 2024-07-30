//! Benchmark for vector search.

mod utils;

use abd_clam::{cakes::OffsetBall, partition::ParPartition, Ball, Cluster, FlatVec, Metric};
use criterion::*;
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

        utils::compare_permuted(
            c,
            "vector-search",
            metric_name,
            &data,
            &root,
            &perm_data,
            &perm_root,
            &queries,
            &radii,
            &ks,
            false,
        );
    }
}

criterion_group!(benches, vector_search);
criterion_main!(benches);
