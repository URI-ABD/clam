#![allow(missing_docs)]

use symagen::random_data;
use test_case::test_case;

use distances::{
    blas, simd,
    vectors::{cosine, dot_product, euclidean, euclidean_sq},
};

#[test_case(euclidean_sq, simd::euclidean_sq_f32, None, 10_f32; "euclidean_sq_f32")]
#[test_case(euclidean, simd::euclidean_f32, Some(blas::euclidean_f32), 10_f32; "euclidean_f32")]
#[test_case(cosine, simd::cosine_f32, Some(blas::cosine_f32), 1_f32; "cosine_f32")]
#[test_case(dot_product, simd::dot_product_f32, None, 1_f32; "dot_product_f32")]
fn simd_distances_f32(naive_fn: fn(&[f32], &[f32]) -> f32, simd_fn: fn(&[f32], &[f32]) -> f32, blas_fn: Option<fn(&[f32], &[f32]) -> f32>, limit: f32) {
    let (cardinality, dimensionality) = (100, 2_usize.pow(12));

    let limit = limit.abs();
    let (min_val, max_val) = (-limit, limit);

    let mut rng = rand::rng();

    let data_x = random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rng);
    let data_y = random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rng);
    let mut failures = Vec::new();

    for (i, x) in data_x.iter().enumerate() {
        for (j, y) in data_y.iter().enumerate() {
            let expected = naive_fn(x, y);

            let actual = simd_fn(x, y);
            let delta = (expected - actual).abs();
            let threshold = f32::EPSILON.sqrt() * actual;
            if delta > threshold {
                failures.push(("simd", i, j, delta, threshold, actual, expected));
            }

            if let Some(blas_fn) = blas_fn {
                let actual = blas_fn(x, y);
                let delta = (expected - actual).abs();
                let threshold = f32::EPSILON.sqrt() * actual;
                if delta > threshold {
                    failures.push(("blas", i, j, delta, threshold, actual, expected));
                }
            }
        }
    }

    assert!(failures.is_empty(), "{} non-empty failures, {:?} ...", failures.len(), &failures[..5]);
}

#[test_case(euclidean_sq, simd::euclidean_sq_f64, None, 10_f64; "euclidean_sq_f64")]
#[test_case(euclidean, simd::euclidean_f64, Some(blas::euclidean_f64), 10_f64; "euclidean_f64")]
#[test_case(cosine, simd::cosine_f64, Some(blas::cosine_f64), 1_f64; "cosine_f64")]
#[test_case(dot_product, simd::dot_product_f64, None, 1_f64; "dot_product_f64")]
fn simd_distances_f64(naive_fn: fn(&[f64], &[f64]) -> f64, simd_fn: fn(&[f64], &[f64]) -> f64, blas_fn: Option<fn(&[f64], &[f64]) -> f64>, limit: f64) {
    let (cardinality, dimensionality) = (100, 2_usize.pow(12));

    let limit = limit.abs();
    let (min_val, max_val) = (-limit, limit);

    let mut rng = rand::rng();

    let data_x = random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rng);
    let data_y = random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rng);
    let mut failures = Vec::new();

    for (i, x) in data_x.iter().enumerate() {
        for (j, y) in data_y.iter().enumerate() {
            let expected = naive_fn(x, y);

            let actual = simd_fn(x, y);
            let delta = (expected - actual).abs();
            let threshold = f64::EPSILON.sqrt() * actual;
            if delta > threshold {
                failures.push(("simd", i, j, delta, threshold, actual, expected));
            }

            if let Some(blas_fn) = blas_fn {
                let actual = blas_fn(x, y);
                let delta = (expected - actual).abs();
                let threshold = f64::EPSILON.sqrt() * actual;
                if delta > threshold {
                    failures.push(("blas", i, j, delta, threshold, actual, expected));
                }
            }
        }
    }

    assert!(failures.is_empty(), "{} non-empty failures, {:?} ...", failures.len(), &failures[..5]);
}
