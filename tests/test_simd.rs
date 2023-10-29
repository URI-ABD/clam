use symagen::random_data;
use test_case::test_case;

use distances::{
    simd,
    vectors::{cosine, euclidean, euclidean_sq},
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

#[test_case(euclidean_sq, simd::euclidean_sq_f32; "euclidean_sq_f32")]
#[test_case(euclidean, simd::euclidean_f32; "euclidean_f32")]
#[test_case(cosine, simd::cosine_f32; "cosine_f32")]
fn simd_distances_f32(naive: fn(&[f32], &[f32]) -> f32, simd: fn(&[f32], &[f32]) -> f32) {
    let data_x = gen_data_f32(42);
    let data_y = gen_data_f32(65);
    let mut failures = Vec::new();

    for (i, x) in data_x.iter().enumerate() {
        for (j, y) in data_y.iter().enumerate() {
            let expected: f32 = naive(x, y);
            let actual: f32 = simd(x, y);
            let delta = (expected - actual).abs();
            let threshold = 1e-5 * actual;
            if delta > threshold {
                failures.push((i, j, delta, threshold));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} non-empty failures, {:?} ...",
        failures.len(),
        &failures[..5]
    );
}

#[test_case(euclidean_sq, simd::euclidean_sq_f64; "euclidean_sq_f64")]
#[test_case(euclidean, simd::euclidean_f64; "euclidean_f64")]
#[test_case(cosine, simd::cosine_f64; "cosine_f64")]
fn simd_distances_f64(naive: fn(&[f64], &[f64]) -> f64, simd: fn(&[f64], &[f64]) -> f64) {
    let data_x = gen_data_f64(42);
    let data_y = gen_data_f64(65);
    let mut failures = Vec::new();

    for (i, x) in data_x.iter().enumerate() {
        for (j, y) in data_y.iter().enumerate() {
            let expected: f64 = naive(x, y);
            let actual: f64 = simd(x, y);
            let delta = (expected - actual).abs();
            let threshold = 1e-10 * actual;
            if delta > threshold {
                failures.push((i, j, delta, threshold));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} non-empty failures, {:?} ...",
        failures.len(),
        &failures[..5]
    );
}
