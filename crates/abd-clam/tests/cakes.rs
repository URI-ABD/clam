//! Tests of the CAKES algorithms.

use std::fmt::Debug;

use abd_clam::{
    cakes::{KnnBreadthFirst, KnnDepthFirst, KnnRepeatedRnn, PermutedBall, RnnClustered},
    Ball, Cluster, DistanceValue, ParCluster, ParDataset, ParPartition, Partition, Permutable,
};
use rand::prelude::*;
use test_case::test_case;

mod common;

#[test]
fn line() {
    let data = common::data_gen::gen_line_data(10);
    let metric = common::metrics::absolute_difference;
    let query = &0;
    let criteria = |c: &Ball<_>| c.cardinality() > 1;

    let ball = Ball::new_tree(&data, &metric, &criteria);
    let par_ball = Ball::par_new_tree(&data, &metric, &criteria);

    let mut perm_data = data.clone();
    let (perm_ball, _) = PermutedBall::par_from_cluster_tree(par_ball.clone(), &mut perm_data);

    for radius in 0..=4 {
        let alg = RnnClustered(radius);
        common::search::check_rnn(&ball, &data, &metric, query, radius, &alg);
        common::search::check_rnn(&par_ball, &data, &metric, query, radius, &alg);
    }

    for k in [1, 4, 8] {
        let alg = KnnDepthFirst(k);
        common::search::check_knn(&ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&par_ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&perm_ball, &perm_data, &metric, query, k, &alg);

        let alg = KnnBreadthFirst(k);
        common::search::check_knn(&ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&par_ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&perm_ball, &perm_data, &metric, query, k, &alg);

        let alg = KnnRepeatedRnn(k, 2);
        common::search::check_knn(&ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&par_ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&perm_ball, &perm_data, &metric, query, k, &alg);
    }
}

#[test]
fn grid() {
    let data = common::data_gen::gen_grid_data(10);
    let metric = common::metrics::hypotenuse;
    let query = &(0.0, 0.0);
    let criteria = |c: &Ball<f32>| c.cardinality() > 1;

    let ball = Ball::new_tree(&data, &metric, &criteria);
    let par_ball = Ball::par_new_tree(&data, &metric, &criteria);

    let mut perm_data = data.clone();
    let (perm_ball, _) = PermutedBall::par_from_cluster_tree(par_ball.clone(), &mut perm_data);

    for radius in 0..=4 {
        let radius = radius as f32;
        let alg = RnnClustered(radius);
        common::search::check_rnn(&ball, &data, &metric, query, radius, &alg);
        common::search::check_rnn(&par_ball, &data, &metric, query, radius, &alg);
    }

    for k in [1, 10] {
        let alg = KnnDepthFirst(k);
        common::search::check_knn(&ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&par_ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&perm_ball, &perm_data, &metric, query, k, &alg);

        let alg = KnnBreadthFirst(k);
        common::search::check_knn(&ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&par_ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&perm_ball, &perm_data, &metric, query, k, &alg);

        let alg = KnnRepeatedRnn(k, 2.0);
        common::search::check_knn(&ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&par_ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&perm_ball, &perm_data, &metric, query, k, &alg);
    }
}

#[test_case(1_000, 10)]
#[test_case(10_000, 10)]
#[test_case(1_000, 100)]
#[test_case(10_000, 100)]
fn vectors(car: usize, dim: usize) {
    let seed = 42;
    let data = {
        let max = 10.0;
        let mut rng = StdRng::seed_from_u64(seed);
        symagen::random_data::random_tabular(car, dim, -max, max, &mut rng)
    };
    let query = vec![0.0; dim];
    let criteria = |c: &Ball<_>| c.cardinality() > 1;

    let radii = [0.01, 0.1];
    let ks = [1, 10];

    let metrics: Vec<Box<dyn (Fn(&Vec<f64>, &Vec<f64>) -> f64) + Send + Sync>> = vec![
        Box::new(common::metrics::euclidean),
        Box::new(common::metrics::manhattan),
    ];

    for metric in &metrics {
        let data = data.clone();

        let ball = Ball::new_tree(&data, metric, &criteria);
        let par_ball = Ball::par_new_tree(&data, metric, &criteria);

        let mut perm_data = data.clone();
        let (perm_ball, _) = PermutedBall::par_from_cluster_tree(par_ball.clone(), &mut perm_data);

        let radii = radii.iter().map(|&f| f * ball.radius()).collect::<Vec<_>>();

        build_and_check_search(
            (&ball, &data),
            &par_ball,
            (&perm_ball, &perm_data),
            metric,
            &query,
            &radii,
            &ks,
        );
    }
}

#[test_case(16, 16, 2)]
#[test_case(32, 16, 3)]
fn strings(num_clumps: usize, clump_size: usize, clump_radius: u16) -> Result<(), String> {
    let seed_length = 30;
    let alphabet = "ACTGN".chars().collect::<Vec<_>>();
    let seed_string = symagen::random_edits::generate_random_string(seed_length, &alphabet);
    let penalties = distances::strings::Penalties::default();
    let inter_clump_distance_range = (clump_radius * 5, clump_radius * 7);
    let len_delta = seed_length / 10;
    let (_, sequences) = symagen::random_edits::generate_clumped_data(
        &seed_string,
        penalties,
        &alphabet,
        num_clumps,
        clump_size,
        clump_radius,
        inter_clump_distance_range,
        len_delta,
    )
    .into_iter()
    .unzip::<_, _, Vec<_>, Vec<_>>();

    let query = &seed_string;

    let radii = [0.01, 0.1];
    let ks = [1, 10];

    let metric = common::metrics::levenshtein;
    let criteria = |c: &Ball<_>| c.cardinality() > 1;

    let ball = Ball::new_tree(&sequences, &metric, &criteria);
    let par_ball = Ball::par_new_tree(&sequences, &metric, &criteria);

    let mut perm_sequences = sequences.clone();
    let (perm_ball, _) = PermutedBall::par_from_cluster_tree(par_ball.clone(), &mut perm_sequences);

    let radii = radii
        .iter()
        .map(|&f| f * (ball.radius() as f32))
        .map(|r| r.ceil() as usize)
        .collect::<Vec<_>>();

    build_and_check_search(
        (&ball, &sequences),
        &par_ball,
        (&perm_ball, &perm_sequences),
        &metric,
        &query,
        &radii,
        &ks,
    );

    Ok(())
}

/// Build trees and check the search results.
fn build_and_check_search<I, T, C, M, D, Pd>(
    ball_data: (&C, &D),
    par_ball: &C,
    perm_ball_data: (&PermutedBall<T, C>, &Pd),
    metric: &M,
    query: &I,
    radii: &[T],
    ks: &[usize],
) where
    I: Debug + Send + Sync + Clone,
    T: DistanceValue + Send + Sync + Debug,
    C: ParCluster<T>,
    M: (Fn(&I, &I) -> T) + Send + Sync,
    D: ParDataset<I> + Permutable<I> + Clone,
    Pd: ParDataset<I> + Clone,
{
    let (ball, data) = ball_data;
    let (perm_ball, perm_data) = perm_ball_data;

    for &radius in radii {
        let alg = RnnClustered(radius);
        common::search::check_rnn(ball, data, metric, query, radius, &alg);
        common::search::check_rnn(par_ball, data, metric, query, radius, &alg);
        common::search::check_rnn(perm_ball, perm_data, metric, query, radius, &alg);
    }

    for &k in ks {
        let alg = KnnRepeatedRnn(k, T::one() + T::one());
        common::search::check_knn(ball, data, metric, query, k, &alg);
        common::search::check_knn(par_ball, data, metric, query, k, &alg);
        common::search::check_knn(perm_ball, perm_data, metric, query, k, &alg);

        let alg = KnnBreadthFirst(k);
        common::search::check_knn(ball, data, metric, query, k, &alg);
        common::search::check_knn(par_ball, data, metric, query, k, &alg);
        common::search::check_knn(perm_ball, perm_data, metric, query, k, &alg);

        let alg = KnnDepthFirst(k);
        common::search::check_knn(ball, data, metric, query, k, &alg);
        common::search::check_knn(par_ball, data, metric, query, k, &alg);
        common::search::check_knn(perm_ball, perm_data, metric, query, k, &alg);
    }
}
