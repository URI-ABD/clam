// use core::f32::EPSILON;

use symagen::random_data;

use distances::simd;

// fn l1_f32(x: &[f32], y: &[f32]) -> f32 {
//     x.iter()
//         .zip(y.iter())
//         .fold(0., |acc, (x, y)| acc + (x - y).abs())
// }

fn l2_sq_f32(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .fold(0., |acc, (x, y)| acc + (x - y).powi(2))
}

fn l2_f32(x: &[f32], y: &[f32]) -> f32 {
    l2_sq_f32(x, y).sqrt()
}

// fn l1_f64(x: &[f64], y: &[f64]) -> f64 {
//     x.iter()
//         .zip(y.iter())
//         .fold(0., |acc, (x, y)| acc + (x - y).abs())
// }

fn l2_sq_f64(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .fold(0., |acc, (x, y)| acc + (x - y).powi(2))
}

fn l2_f64(x: &[f64], y: &[f64]) -> f64 {
    l2_sq_f64(x, y).sqrt()
}

fn cos_f32(x: &[f32], y: &[f32]) -> f32 {
    let [xx, yy, xy] = x
        .iter()
        .zip(y.iter())
        .fold([0_f32; 3], |[xx, yy, xy], (&a, &b)| {
            [a.mul_add(a, xx), b.mul_add(b, yy), a.mul_add(b, xy)]
        });

    if xx < f32::EPSILON || yy < f32::EPSILON || xy < f32::EPSILON {
        1_f32
    } else {
        let d = 1_f32 - xy / (xx * yy).sqrt();
        if d < f32::EPSILON {
            0_f32
        } else {
            d
        }
    }
}

fn cos_f64(x: &[f64], y: &[f64]) -> f64 {
    let [xx, yy, xy] = x
        .iter()
        .zip(y.iter())
        .fold([0_f64; 3], |[xx, yy, xy], (&a, &b)| {
            [a.mul_add(a, xx), b.mul_add(b, yy), a.mul_add(b, xy)]
        });

    if xx < f64::EPSILON || yy < f64::EPSILON || xy < f64::EPSILON {
        1_f64
    } else {
        let d = 1_f64 - xy / (xx * yy).sqrt();
        if d < f64::EPSILON {
            0_f64
        } else {
            d
        }
    }
}

fn gen_data_f32() -> Vec<Vec<f32>> {
    let seed = 4;
    let (cardinality, dimensionality) = (100, 10_000);
    let (min_val, max_val) = (-10., 10.);

    random_data::random_f32(cardinality, dimensionality, min_val, max_val, seed)
}

fn gen_data_f64() -> Vec<Vec<f64>> {
    let seed = 4;
    let (cardinality, dimensionality) = (100, 10_000);
    let (min_val, max_val) = (-10., 10.);

    random_data::random_f64(cardinality, dimensionality, min_val, max_val, seed)
}

#[test]
fn lp_f32_simd() {
    let data = gen_data_f32();
    let mut count = 0;
    for x in data.iter() {
        for y in data.iter() {
            count += 1;
            let expected = l2_sq_f32(x, y);
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

            let expected = l2_f32(x, y);
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
fn cos_f32_simd() {
    let data = gen_data_f32();
    let mut count = 0;
    for x in data.iter() {
        for y in data.iter() {
            count += 1;
            let expected = cos_f32(x, y);
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
fn lp_f64_simd() {
    let data = gen_data_f64();
    let mut count = 0;
    for x in data.iter() {
        for y in data.iter() {
            count += 1;
            let expected = l2_sq_f64(x, y);
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

            let expected = l2_f64(x, y);
            let actual: f64 = simd::euclidean_f64(x, y);
            assert!(
                (expected - actual).abs() <= 8.0 * f64::EPSILON,
                "SIMD Euclidean #{count}: expected: {}, actual: {}",
                expected,
                actual
            );
        }
    }
}

#[test]
fn cos_f64_simd() {
    let data = gen_data_f64();
    let mut count = 0;
    for x in data.iter() {
        for y in data.iter() {
            count += 1;
            let expected = cos_f64(x, y);
            let actual: f64 = simd::cosine_f64(x, y);
            assert!(
                (expected - actual).abs() <= 8.0 * f64::EPSILON,
                "SIMD Cosine #{count}: expected: {expected}, actual: {actual}"
            );
        }
    }
}
