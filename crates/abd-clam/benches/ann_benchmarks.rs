//! Benchmarks for the suite of ANN-Benchmarks datasets.

mod utils;

use abd_clam::{
    adapter::{Adapter, ParBallAdapter},
    cakes::OffBall,
    partition::ParPartition,
    BalancedBall, Ball, Cluster, Metric, Permutable,
};
use criterion::*;

fn ann_benchmarks(c: &mut Criterion) {
    let root_str = std::env::var("ANN_DATA_ROOT").unwrap();
    let ann_data_root = std::path::Path::new(&root_str).canonicalize().unwrap();
    println!("ANN data root: {:?}", ann_data_root);

    let euclidean = |x: &Vec<_>, y: &Vec<_>| distances::vectors::euclidean(x, y);
    let cosine = |x: &Vec<_>, y: &Vec<_>| distances::vectors::cosine(x, y);
    let data_names: Vec<(&str, &str, Metric<Vec<f32>, f32>)> = vec![
        ("fashion-mnist", "euclidean", Metric::new(euclidean, false)),
        ("glove-25", "cosine", Metric::new(cosine, false)),
        ("sift", "euclidean", Metric::new(euclidean, false)),
    ];

    let data_pairs = data_names.into_iter().map(|(data_name, metric_name, metric)| {
        (
            data_name,
            metric_name,
            utils::read_ann_data_npy(data_name, &ann_data_root, metric),
        )
    });

    let seed = Some(42);
    let radii = vec![];
    let ks = vec![10, 100];
    let num_queries = 100;
    for (data_name, metric_name, (data, queries)) in data_pairs {
        let queries = &queries[0..num_queries];

        let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
        let ball = Ball::par_new_tree(&data, &criteria, seed);
        let (off_ball, perm_data) = OffBall::par_from_ball_tree(ball.clone(), data.clone());

        let criteria = |c: &BalancedBall<_, _, _>| c.cardinality() > 1;
        let balanced_ball = BalancedBall::par_new_tree(&data, &criteria, seed);
        let (balanced_off_ball, balanced_perm_data) = {
            let (balanced_off_ball, permutation) = OffBall::adapt_tree(balanced_ball.clone(), None);
            let mut balanced_perm_data = data.clone();
            balanced_perm_data.permute(&permutation);
            (balanced_off_ball, balanced_perm_data)
        };

        utils::compare_permuted(
            c,
            data_name,
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
            true,
        );
    }
}

criterion_group!(benches, ann_benchmarks);
criterion_main!(benches);
