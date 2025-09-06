#![allow(missing_docs)]

use rand::prelude::*;
use symagen::random_data;

use distances::vectors::{chebyshev, dot_product, euclidean, euclidean_sq, l3_norm, l4_norm, manhattan, pearson};

fn l1(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).fold(0., |acc, (x, y)| acc + (x - y).abs())
}

fn l2_sq(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).fold(0., |acc, (x, y)| acc + (x - y).powi(2))
}

fn l2(x: &[f32], y: &[f32]) -> f32 {
    l2_sq(x, y).sqrt()
}

fn l3(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).fold(0., |acc, (x, y)| acc + (x - y).abs().powi(3)).cbrt()
}

fn l4(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).fold(0., |acc, (x, y)| acc + (x - y).powi(4)).sqrt().sqrt()
}

fn l_inf(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).fold(0., |acc, (x, y)| acc.max((x - y).abs()))
}

fn dot(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).fold(0., |acc, (x, y)| acc + x * y)
}

#[test]
fn lp_f32() {
    let seed = 42;
    let (cardinality, dimensionality) = (100, 10_000);
    let (min_val, max_val) = (-1.0, 1.0);

    let data = random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rand::rngs::StdRng::seed_from_u64(seed));

    for x in data.iter() {
        for y in data.iter() {
            let e_l1 = l1(x, y);
            let a_l1: f32 = manhattan(x, y);
            assert!((e_l1 - a_l1).abs() <= f32::EPSILON, "Manhattan: expected: {}, actual: {}", e_l1, a_l1);

            let expected = l2_sq(x, y);
            let actual: f32 = euclidean_sq(x, y);
            assert!(
                (expected - actual).abs() <= f32::EPSILON,
                "Euclidean squared: expected: {}, actual: {}",
                expected,
                actual
            );

            let e_l2 = l2(x, y);
            let a_l2: f32 = euclidean(x, y);
            assert!((e_l2 - a_l2).abs() <= f32::EPSILON, "Euclidean: expected: {}, actual: {}", e_l2, a_l2);

            let e_l3 = l3(x, y);
            let a_l3: f32 = l3_norm(x, y);
            assert!((e_l3 - a_l3).abs() <= f32::EPSILON, "L3 norm: expected: {}, actual: {}", e_l3, a_l3);

            let e_l4 = l4(x, y);
            let a_l4: f32 = l4_norm(x, y);
            assert!((e_l4 - a_l4).abs() <= f32::EPSILON, "L4 norm: expected: {}, actual: {}", e_l4, a_l4);

            let e_l_inf = l_inf(x, y);
            let a_l_inf: f32 = chebyshev(x, y);
            assert!(
                (e_l_inf - a_l_inf).abs() <= f32::EPSILON,
                "Chebyshev: expected: {}, actual: {}",
                e_l_inf,
                a_l_inf
            );

            // We allow a bit more slack for dot product due to greater
            // accumulation of floating point errors with larger float values
            let e_dot = dot(&x, &y);
            let a_dot: f32 = dot_product(&x, &y);
            assert!(
                (e_dot - a_dot).abs() / (e_dot * e_dot) <= f32::EPSILON,
                "Dot product: expected: {}, actual: {}",
                e_dot,
                a_dot
            );
        }
    }
}

#[test]
fn pearson_test() {
    let seed = 42;
    let (cardinality, dimensionality) = (100, 10_000);
    let (min_val, max_val) = (-10., 10.);

    let data_1 = random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rand::rngs::StdRng::seed_from_u64(seed));

    let data_2 = random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rand::rngs::StdRng::seed_from_u64(seed + 1));

    for x in data_1.iter() {
        for y in data_2.iter() {
            // Basic Pearson tests

            // Two different sets
            let (p_lb, p_ub): (f32, f32) = (0.0, 2.0);
            let actual: f32 = pearson(&x, &y);
            assert!(
                p_lb - f32::EPSILON <= actual && actual <= p_ub + f32::EPSILON,
                "Pearson basic: expected range: ({}, {}), actual: {}",
                p_lb,
                p_ub,
                actual
            );

            // Perfect positive correlation
            let expected: f32 = 0.0;
            let actual: f32 = pearson(&x, &x);
            assert!(
                (expected - actual).abs() <= f32::EPSILON,
                "Pearson positive: expected: {}, actual: {}",
                expected,
                actual
            );

            // No correlation
            let p1: [f32; 4] = [1.0, 1.0, -1.0, -1.0];
            let p2: [f32; 4] = [1.0, -1.0, 1.0, -1.0];
            let expected: f32 = 1.0;
            let actual: f32 = pearson(&p1, &p2);
            assert!(
                (expected - actual).abs() <= f32::EPSILON,
                "Pearson zero: expected: {}, actual: {}",
                expected,
                actual
            );

            // Perfect negative correlation (with slope of -8)
            // Note: Test fails unless slope is a square of 2,
            // likely due to multiplication hitting limits of f32 precision
            let x_inv: Vec<f32> = x.iter().map(|&n| n * -8.0).collect();
            let expected: f32 = 2.0;
            let actual: f32 = pearson(&x, &x_inv);
            assert!(
                (expected - actual).abs() <= f32::EPSILON,
                "Pearson negative: expected: {}, actual: {}",
                expected,
                actual
            );
        }
    }
}
