use core::f32::EPSILON;

use rand::prelude::*;
use symagen::random_data;

use distances::{
    vectors::{chebyshev, euclidean, euclidean_sq, l3_norm, l4_norm, manhattan},
    Number,
};

fn l1(x: &[u32], y: &[u32]) -> u32 {
    x.iter().zip(y.iter()).map(|(x, &y)| x.abs_diff(y)).sum()
}

fn l2_sq(x: &[u32], y: &[u32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(x, &y)| x.abs_diff(y).as_f32())
        .map(|v| v * v)
        .sum()
}

fn l2(x: &[u32], y: &[u32]) -> f32 {
    l2_sq(x, y).sqrt()
}

fn l3(x: &[u32], y: &[u32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(x, &y)| x.abs_diff(y).as_f32())
        .map(|v| v * v * v)
        .sum::<f32>()
        .cbrt()
}

fn l4(x: &[u32], y: &[u32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(x, &y)| x.abs_diff(y).as_f32())
        .map(|v| v * v)
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt()
        .sqrt()
}

fn l_inf(x: &[u32], y: &[u32]) -> u32 {
    x.iter()
        .zip(y.iter())
        .map(|(x, &y)| x.abs_diff(y))
        .max()
        .unwrap()
}

#[test]
fn lp_u32() {
    let seed = 42;
    let (cardinality, dimensionality) = (100, 10_000);
    let (min_val, max_val) = (0, 10_000);

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
            let a_l1 = manhattan(x, y);
            assert_eq!(e_l1, a_l1, "Manhattan: expected: {e_l1}, actual: {a_l1}");

            let e_l2s = l2_sq(x, y);
            let a_l2s: f32 = euclidean_sq(x, y);
            assert!(
                (e_l2s - a_l2s).abs() <= EPSILON,
                "Euclidean squared: expected: {e_l2s}, actual: {a_l2s}"
            );

            let e_l2 = l2(x, y);
            let a_l2: f32 = euclidean(x, y);
            assert!(
                (e_l2 - a_l2).abs() <= EPSILON,
                "Euclidean: expected: {e_l2}, actual: {a_l2}"
            );

            let e_l3 = l3(x, y);
            let a_l3: f32 = l3_norm(x, y);
            assert!(
                (e_l3 - a_l3).abs() <= EPSILON,
                "L3 norm: expected: {e_l3}, actual: {a_l3}"
            );

            let e_l4 = l4(x, y);
            let a_l4: f32 = l4_norm(x, y);
            assert!(
                (e_l4 - a_l4).abs() <= EPSILON,
                "L4 norm: expected: {e_l4}, actual: {a_l4}"
            );

            let e_l_inf = l_inf(x, y);
            let a_l_inf = chebyshev(x, y);
            assert_eq!(
                e_l_inf, a_l_inf,
                "Chebyshev: expected: {e_l_inf}, actual: {a_l_inf}",
            );
        }
    }
}
