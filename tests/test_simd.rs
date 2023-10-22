// use core::f32::EPSILON;

use symagen::random_data;

use distances::simd;

use distances::vectors::{cosine, euclidean, euclidean_sq};

use distances::Number;
// fn l1_f32(x: &[f32], y: &[f32]) -> f32 {
//     x.iter()
//         .zip(y.iter())
//         .fold(0., |acc, (x, y)| acc + (x - y).abs())
// }

// fn l2_sq_f32(x: &[f32], y: &[f32]) -> f32 {
//     x.iter()
//         .zip(y.iter())
//         .fold(0., |acc, (x, y)| acc + (x - y).powi(2))
// }
//
// fn l2_f32(x: &[f32], y: &[f32]) -> f32 {
//     l2_sq_f32(x, y).sqrt()
// }
//
// // fn l1_f64(x: &[f64], y: &[f64]) -> f64 {
// //     x.iter()
// //         .zip(y.iter())
// //         .fold(0., |acc, (x, y)| acc + (x - y).abs())
// // }
//
// fn l2_sq_f64(x: &[f64], y: &[f64]) -> f64 {
//     x.iter()
//         .zip(y.iter())
//         .fold(0., |acc, (x, y)| acc + (x - y).powi(2))
// }
//
// fn l2_f64(x: &[f64], y: &[f64]) -> f64 {
//     l2_sq_f64(x, y).sqrt()
// }

// fn cos_f32(x: &[f32], y: &[f32]) -> f32 {
//     let [xx, yy, xy] = x
//         .iter()
//         .zip(y.iter())
//         .fold([0_f32; 3], |[xx, yy, xy], (&a, &b)| {
//             [a.mul_add(a, xx), b.mul_add(b, yy), a.mul_add(b, xy)]
//         });
//
//     if xx < f32::EPSILON || yy < f32::EPSILON || xy < f32::EPSILON {
//         1_f32
//     } else {
//         let d = 1_f32 - xy / (xx * yy).sqrt();
//         if d < f32::EPSILON {
//             0_f32
//         } else {
//             d
//         }
//     }
// }
//
// fn cos_f64(x: &[f64], y: &[f64]) -> f64 {
//     let [xx, yy, xy] = x
//         .iter()
//         .zip(y.iter())
//         .fold([0_f64; 3], |[xx, yy, xy], (&a, &b)| {
//             [a.mul_add(a, xx), b.mul_add(b, yy), a.mul_add(b, xy)]
//         });
//
//     if xx < f64::EPSILON || yy < f64::EPSILON || xy < f64::EPSILON {
//         1_f64
//     } else {
//         let d = 1_f64 - xy / (xx * yy).sqrt();
//         if d < f64::EPSILON {
//             0_f64
//         } else {
//             d
//         }
//     }
// }

fn gen_data_f32() -> Vec<Vec<f32>> {
    let seed = 42;
    let (cardinality, dimensionality) = (100, 10_000);
    let (min_val, max_val) = (-10., 10.);

    random_data::random_f32(cardinality, dimensionality, min_val, max_val, seed)
}

fn gen_data_f64(seed: u64) -> Vec<Vec<f64>> {
    // let seed = 42;
    let (cardinality, dimensionality) = (100, 2_usize.pow(12));
    let (min_val, max_val) = (-10., 10.);

    random_data::random_f64(cardinality, dimensionality, min_val, max_val, seed)
}

#[test]
#[ignore]
fn lp_squared_f32_simd() {
    let data = gen_data_f32();
    let mut count = 0;
    for x in data.iter() {
        for y in data.iter() {
            count += 1;
            let expected: f32 = euclidean_sq(x, y);
            let actual: f32 = simd::euclidean_sq_f32(x, y);
            assert!(
                (expected - actual).abs() <= 48.0 * f32::EPSILON,
                "SIMD Euclidean squared #{count}: expected: {}, actual: {}\nx: {:?}\ny: {:?}\ne:{}",
                expected,
                actual,
                x,
                y,
                48.0 * f32::EPSILON
            );
        }
    }
}

#[test]
#[ignore]
fn lp_f32_simd() {
    let data = gen_data_f32();
    let mut count = 0;
    for x in data.iter() {
        for y in data.iter() {
            count += 1;

            let expected: f32 = euclidean(x, y);
            let actual: f32 = simd::euclidean_f32(x, y);
            assert!(
                (expected - actual).abs() <= 8.0 * f32::EPSILON,
                "SIMD Euclidean: expected #{count}: {}, actual: {}\nx: {:?}\ny: {:?}",
                expected,
                actual,
                x,
                y
            );
        }
    }
}

#[test]
#[ignore]
fn cos_f32_simd() {
    let data = gen_data_f32();
    let mut count = 0;
    for x in data.iter() {
        for y in data.iter() {
            count += 1;
            let expected: f32 = cosine(x, y);
            let actual: f32 = simd::cosine_f32(x, y);
            assert!(
                (expected - actual).abs() <= 8.0 * f32::EPSILON,
                "SIMD Cosine: expected #{count}: {}, actual: {}\nx: {:?}\ny: {:?}",
                expected,
                actual,
                x,
                y
            );
        }
    }
}

#[test]
#[ignore]
fn lp_squared_f64_simd() {
    let data = gen_data_f64(42);
    let mut count = 0;
    for x in data.iter() {
        for y in data.iter() {
            count += 1;
            let expected: f64 = euclidean_sq(x, y);
            let actual: f64 = simd::euclidean_sq_f64(x, y);
            assert!(
                (expected - actual).abs() <= 48.0 * f64::EPSILON,
                "SIMD Euclidean squared #{count}: expected: {}, actual: {}\nx: {:?}\ny: {:?}\ne:{}",
                expected,
                actual,
                x,
                y,
                48.0 * f64::EPSILON
            );
        }
    }
}

#[test]

fn lp_f64_simd() {
    let datax = gen_data_f64(42);
    let datay = gen_data_f64(65);
    let logdim = 1.0 + datax[0].len().as_f64().log2();
    // let mut count = 0;
    // let mut failures = 0;
    let mut failures = Vec::new();
    for (i, x) in datax.iter().enumerate() {
        for (j, y) in datay.iter().enumerate() {
            // count += 1;

            let expected: f64 = euclidean(x, y);
            let actual: f64 = simd::euclidean_f64(x, y);
            let delta = (expected - actual).abs();
            let threshold = 2.0 * logdim * actual * f64::EPSILON;
            if delta > threshold {
                failures.push((i, j, delta, threshold));
            }
            // assert!(
            //     (expected - actual).abs() <= 8.0 * f64::EPSILON,
            //     "SIMD Euclidean #{count}: expected: {}, actual: {}",
            //     expected,
            //     actual
            // );
        }
    }
    assert!(
        failures.is_empty(),
        "{} non-empty failures, {:?}",
        failures.len(),
        &failures[..5]
    );
}

#[test]
#[ignore]
fn cos_f64_simd() {
    let data = gen_data_f64(42);
    let mut count = 0;
    for x in data.iter() {
        for y in data.iter() {
            count += 1;
            let expected: f64 = cosine(x, y);
            let actual: f64 = simd::cosine_f64(x, y);
            assert!(
                (expected - actual).abs() <= 8.0 * f64::EPSILON,
                "SIMD Cosine #{count}: expected: {expected}, actual: {actual}"
            );
        }
    }
}
