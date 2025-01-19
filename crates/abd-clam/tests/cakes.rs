//! Tests of the CAKES algorithms.

use std::collections::HashMap;

use distances::Number;
use test_case::test_case;

use abd_clam::{
    cakes::{self, HintedDataset, ParHintedDataset, PermutedBall},
    cluster::{
        adapter::{BallAdapter, ParBallAdapter},
        ParCluster, ParPartition, Partition,
    },
    dataset::{AssociatesMetadataMut, Permutable},
    metric::{AbsoluteDifference, Euclidean, Hypotenuse, Levenshtein, Manhattan, ParMetric},
    Ball, Cluster, Dataset, FlatVec,
};

mod common;

#[test]
fn line() {
    let data = common::data_gen::gen_line_data(10).transform_metadata(|&i| (i, HashMap::<usize, i32>::new()));
    let metric = AbsoluteDifference;
    let query = &0;
    let criteria = |c: &Ball<_>| c.cardinality() > 1;
    let seed = Some(42);

    let ball = Ball::new_tree(&data, &metric, &criteria, seed);
    let par_ball = Ball::par_new_tree(&data, &metric, &criteria, seed);

    let data = data.with_hints_from(&metric, &par_ball, 2, 4);
    let (perm_ball, perm_data) = PermutedBall::from_ball_tree(ball.clone(), data.clone(), &metric);

    for radius in 0..=4 {
        let alg = cakes::RnnClustered(radius);
        common::search::check_rnn(&ball, &data, &metric, query, radius, &alg);
        common::search::check_rnn(&par_ball, &data, &metric, query, radius, &alg);
    }

    for k in [1, 4, 8] {
        let alg = cakes::KnnDepthFirst(k);
        common::search::check_knn(&ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&par_ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&perm_ball, &perm_data, &metric, query, k, &alg);

        let alg = cakes::KnnBreadthFirst(k);
        common::search::check_knn(&ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&par_ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&perm_ball, &perm_data, &metric, query, k, &alg);

        let alg = cakes::KnnRepeatedRnn(k, 2);
        common::search::check_knn(&ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&par_ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&perm_ball, &perm_data, &metric, query, k, &alg);

        let alg = cakes::KnnHinted(k);
        common::search::check_knn(&ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&par_ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&perm_ball, &perm_data, &metric, query, k, &alg);
    }
}

#[test]
fn grid() {
    let data = common::data_gen::gen_grid_data(10).transform_metadata(|&i| (i, HashMap::new()));
    let metric = Hypotenuse;
    let query = &(0.0, 0.0);
    let criteria = |c: &Ball<f32>| c.cardinality() > 1;
    let seed = Some(42);

    let ball = Ball::new_tree(&data, &metric, &criteria, seed);
    let par_ball = Ball::par_new_tree(&data, &metric, &criteria, seed);
    let data = data.with_hints_from(&metric, &par_ball, 2.0, 4);

    let (perm_ball, perm_data) = PermutedBall::from_ball_tree(ball.clone(), data.clone(), &metric);

    for radius in 0..=4 {
        let radius = radius.as_f32();
        let alg = cakes::RnnClustered(radius);
        common::search::check_rnn(&ball, &data, &metric, query, radius, &alg);
        common::search::check_rnn(&par_ball, &data, &metric, query, radius, &alg);
    }

    for k in [1, 10] {
        let alg = cakes::KnnDepthFirst(k);
        common::search::check_knn(&ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&par_ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&perm_ball, &perm_data, &metric, query, k, &alg);

        let alg = cakes::KnnBreadthFirst(k);
        common::search::check_knn(&ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&par_ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&perm_ball, &perm_data, &metric, query, k, &alg);

        let alg = cakes::KnnRepeatedRnn(k, 2.0);
        common::search::check_knn(&ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&par_ball, &data, &metric, query, k, &alg);
        common::search::check_knn(&perm_ball, &perm_data, &metric, query, k, &alg);

        let alg = cakes::KnnHinted(k);
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
    let data = common::data_gen::gen_random_data(car, dim, 10.0, seed)
        .with_name("random-vectors")
        .transform_metadata(|&i| (i, HashMap::new()));
    let seed = Some(seed);
    let query = vec![0.0; dim];
    let criteria = |c: &Ball<_>| c.cardinality() > 1;

    let radii = [0.01, 0.1];
    let ks = [1, 10];

    let metrics: Vec<Box<dyn ParMetric<Vec<f64>, f64>>> = vec![Box::new(Euclidean), Box::new(Manhattan)];

    for metric in &metrics {
        let data = data.clone();

        let ball = Ball::new_tree(&data, metric, &criteria, seed);
        let par_ball = Ball::par_new_tree(&data, metric, &criteria, seed);
        let (perm_ball, perm_data) = PermutedBall::par_from_ball_tree(par_ball.clone(), data.clone(), metric);

        let radii = radii
            .iter()
            .map(|&f| f * ball.radius().as_f32())
            .map(|r| r.as_f64())
            .collect::<Vec<_>>();

        let data = data.with_hints_from_tree(&ball, metric);

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
    let (metadata, data) = symagen::random_edits::generate_clumped_data(
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

    let data = FlatVec::new(data)?
        .with_metadata(&metadata)?
        .with_name("random-strings")
        .transform_metadata(|s| (s.clone(), HashMap::new()));
    let query = &seed_string;
    let seed = Some(42);

    let radii = [0.01, 0.1];
    let ks = [1, 10];

    let metric = Levenshtein;
    let criteria = |c: &Ball<u32>| c.cardinality() > 1;

    let ball = Ball::new_tree(&data, &metric, &criteria, seed);
    let par_ball = Ball::par_new_tree(&data, &metric, &criteria, seed);
    let (perm_ball, perm_data) = PermutedBall::par_from_ball_tree(par_ball.clone(), data.clone(), &metric);

    let radii = radii
        .iter()
        .map(|&f| f * ball.radius().as_f32())
        .map(|r| r.as_u32())
        .collect::<Vec<_>>();

    let data = data.with_hints_from_tree(&ball, &metric);

    build_and_check_search(
        (&ball, &data),
        &par_ball,
        (&perm_ball, &perm_data),
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
    I: core::fmt::Debug + Send + Sync + Clone,
    T: Number,
    C: ParCluster<T>,
    M: ParMetric<I, T>,
    D: ParHintedDataset<I, T, C, M> + Permutable + Clone,
    Pd: ParHintedDataset<I, T, PermutedBall<T, C>, M> + Permutable + Clone,
{
    let (ball, data) = ball_data;
    let (perm_ball, perm_data) = perm_ball_data;

    for &radius in radii {
        let alg = cakes::RnnClustered(radius);
        common::search::check_rnn(ball, data, metric, query, radius, &alg);
        common::search::check_rnn(par_ball, data, metric, query, radius, &alg);
        common::search::check_rnn(perm_ball, perm_data, metric, query, radius, &alg);
    }

    for &k in ks {
        let alg = cakes::KnnRepeatedRnn(k, T::ONE.double());
        common::search::check_knn(ball, data, metric, query, k, &alg);
        common::search::check_knn(par_ball, data, metric, query, k, &alg);
        common::search::check_knn(perm_ball, perm_data, metric, query, k, &alg);

        let alg = cakes::KnnBreadthFirst(k);
        common::search::check_knn(ball, data, metric, query, k, &alg);
        common::search::check_knn(par_ball, data, metric, query, k, &alg);
        common::search::check_knn(perm_ball, perm_data, metric, query, k, &alg);

        let alg = cakes::KnnDepthFirst(k);
        common::search::check_knn(ball, data, metric, query, k, &alg);
        common::search::check_knn(par_ball, data, metric, query, k, &alg);
        common::search::check_knn(perm_ball, perm_data, metric, query, k, &alg);

        let alg = cakes::KnnHinted(k);
        common::search::check_knn(ball, data, metric, query, k, &alg);
        common::search::check_knn(par_ball, data, metric, query, k, &alg);
        common::search::check_knn(perm_ball, perm_data, metric, query, k, &alg);
    }
}
