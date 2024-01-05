use core::f32::EPSILON;

use rand::prelude::*;
use symagen::random_data;

use distances::vectors::{chebyshev, euclidean, euclidean_sq, l3_norm, l4_norm, manhattan};

fn l1(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .fold(0., |acc, (x, y)| acc + (x - y).abs())
}

fn l2_sq(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .fold(0., |acc, (x, y)| acc + (x - y).powi(2))
}

fn l2(x: &[f32], y: &[f32]) -> f32 {
    l2_sq(x, y).sqrt()
}

fn l3(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .fold(0., |acc, (x, y)| acc + (x - y).abs().powi(3))
        .cbrt()
}

fn l4(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .fold(0., |acc, (x, y)| acc + (x - y).powi(4))
        .sqrt()
        .sqrt()
}

fn l_inf(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .fold(0., |acc, (x, y)| acc.max((x - y).abs()))
}

#[test]
fn lp_f32() {
    let seed = 42;
    let (cardinality, dimensionality) = (100, 10_000);
    let (min_val, max_val) = (-10., 10.);

    let data = random_data::random_tabular(
        cardinality,
        dimensionality,
        min_val,
        max_val,
        &mut rand::rngs::StdRng::seed_from_u64(seed),
    );

    for x in data.iter() {
        for y in data.iter() {
            let e_l1 = l1(x, y);
            let a_l1: f32 = manhattan(x, y);
            assert!(
                (e_l1 - a_l1).abs() <= EPSILON,
                "Manhattan: expected: {}, actual: {}",
                e_l1,
                a_l1
            );

            let expected = l2_sq(x, y);
            let actual: f32 = euclidean_sq(x, y);
            assert!(
                (expected - actual).abs() <= EPSILON,
                "Euclidean squared: expected: {}, actual: {}",
                expected,
                actual
            );

            let expected = l2(x, y);
            let actual: f32 = euclidean(x, y);
            assert!(
                (expected - actual).abs() <= EPSILON,
                "Euclidean: expected: {}, actual: {}",
                expected,
                actual
            );

            let e_l3 = l3(x, y);
            let a_l3: f32 = l3_norm(x, y);
            assert!(
                (e_l3 - a_l3).abs() <= EPSILON,
                "L3 norm: expected: {}, actual: {}",
                e_l3,
                a_l3
            );

            let e_l4 = l4(x, y);
            let a_l4: f32 = l4_norm(x, y);
            assert!(
                (e_l4 - a_l4).abs() <= EPSILON,
                "L4 norm: expected: {}, actual: {}",
                e_l4,
                a_l4
            );

            let e_l_inf = l_inf(x, y);
            let a_l_inf: f32 = chebyshev(x, y);
            assert!(
                (e_l_inf - a_l_inf).abs() <= EPSILON,
                "Chebyshev: expected: {}, actual: {}",
                e_l_inf,
                a_l_inf
            );
        }
    }
}
