use symagen::random_data;
use test_case::test_case;

use distances::{
    number::Float,
    simd,
    vectors::{cosine, euclidean, euclidean_sq},
};

#[test_case(euclidean_sq, simd::euclidean_sq_f32, 10_f32; "euclidean_sq_f32")]
#[test_case(euclidean, simd::euclidean_f32, 10_f32; "euclidean_f32")]
#[test_case(cosine, simd::cosine_f32, 1_f32; "cosine_f32")]
#[test_case(euclidean_sq, simd::euclidean_sq_f64, 10_f64; "euclidean_sq_f64")]
#[test_case(euclidean, simd::euclidean_f64, 10_f64; "euclidean_f64")]
#[test_case(cosine, simd::cosine_f64, 1_f64; "cosine_f64")]
fn simd_distances<T: Float>(naive: fn(&[T], &[T]) -> T, simd: fn(&[T], &[T]) -> T, limit: T) {
    let (cardinality, dimensionality) = (100, 2_usize.pow(12));

    let limit = limit.abs();
    let (min_val, max_val) = (-limit, limit);

    let mut rng = rand::thread_rng();

    let data_x = random_data::random_tabular::<T, _>(
        cardinality,
        dimensionality,
        min_val,
        max_val,
        &mut rng,
    );
    let data_y = random_data::random_tabular::<T, _>(
        cardinality,
        dimensionality,
        min_val,
        max_val,
        &mut rng,
    );
    let mut failures = Vec::new();

    for (i, x) in data_x.iter().enumerate() {
        for (j, y) in data_y.iter().enumerate() {
            let expected = naive(x, y);
            let actual = simd(x, y);
            let delta = (expected - actual).abs();
            let threshold = T::epsilon().sqrt() * actual;
            if delta > threshold {
                failures.push((i, j, delta, threshold, actual, expected));
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
