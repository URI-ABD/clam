#![allow(unused_imports, dead_code)]

use symagen::random_data;

use distances::{
    simd,
    vectors::{cosine, euclidean, euclidean_sq},
    Number,
};

fn gen_data_f32(seed: u64) -> Vec<Vec<f32>> {
    let (cardinality, dimensionality) = (100, 2_usize.pow(12));
    let (min_val, max_val) = (-10., 10.);

    random_data::random_f32(cardinality, dimensionality, min_val, max_val, seed)
}

fn gen_data_f64(seed: u64) -> Vec<Vec<f64>> {
    let (cardinality, dimensionality) = (100, 2_usize.pow(12));
    let (min_val, max_val) = (-10., 10.);

    random_data::random_f64(cardinality, dimensionality, min_val, max_val, seed)
}

#[test]
fn lp_f64_simd() {
    let datax = gen_data_f64(42);
    let datay = gen_data_f64(65);
    let logdim = 1.0 + datax[0].len().as_f64().log2();
    let mut failures = Vec::new();
    for (i, x) in datax.iter().enumerate() {
        for (j, y) in datay.iter().enumerate() {
            let expected: f64 = euclidean(x, y);
            let actual: f64 = simd::euclidean_f64(x, y);
            let delta = (expected - actual).abs();
            let threshold = 2.0 * logdim * actual * f64::EPSILON;
            if delta > threshold {
                failures.push((i, j, delta, threshold));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} non-empty failures, {:?}",
        failures.len(),
        &failures[..5]
    );
}
