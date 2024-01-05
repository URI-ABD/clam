use rand::prelude::*;
use symagen::random_data;

use distances::number::Float;

/// Generates a vec of 10 million random non-negative f32s
fn gen_f32s() -> Vec<Vec<f32>> {
    random_data::random_tabular(
        1,
        10_000_000,
        -1e10_f32,
        1e10,
        &mut rand::rngs::StdRng::seed_from_u64(42),
    )
}

/// Generates a vec of 10 million random non-negative f64s
fn gen_f64s() -> Vec<Vec<f64>> {
    random_data::random_tabular(
        1,
        10_000_000,
        -1e10_f64,
        1e10,
        &mut rand::rngs::StdRng::seed_from_u64(42),
    )
}

#[test]
fn sqrt_f32() {
    let floats = gen_f32s();

    for f in floats[0].iter().map(|f| f.abs()) {
        let expected = f.sqrt();
        let actual = <f32 as Float>::sqrt(f);

        assert!((expected - actual).abs() < 1e-6);
    }
}

#[test]
fn sqrt_f64() {
    let floats = gen_f64s();

    for f in floats[0].iter().map(|f| f.abs()) {
        let expected = f.sqrt();
        let actual = <f64 as Float>::sqrt(f);

        assert!((expected - actual).abs() < 1e-6);
    }
}

#[test]
fn cbrt_f32() {
    let floats = gen_f32s();

    for &f in floats[0].iter() {
        let expected = f.cbrt();
        let actual = <f32 as Float>::cbrt(f);

        assert!((expected - actual).abs() < 1e-6);
    }
}

#[test]
fn cbrt_f64() {
    let floats = gen_f64s();

    for &f in floats[0].iter() {
        let expected = f.cbrt();
        let actual = <f64 as Float>::cbrt(f);

        assert!((expected - actual).abs() < 1e-6);
    }
}

#[test]
fn inv_sqrt_f32() {
    let floats = gen_f32s();

    for f in floats[0].iter().map(|f| f.abs()) {
        let expected = 1.0 / f.sqrt();
        let actual = <f32 as Float>::inv_sqrt(f);

        if expected > 0. && expected.is_finite() {
            let error = (expected - actual).abs() / expected;
            assert!(error < 1e-6, "{expected} != {actual}");
        } else {
            assert!(actual.is_nan());
        }
    }
}

#[test]
fn inv_sqrt_f64() {
    let floats = gen_f64s();

    for f in floats[0].iter().map(|f| f.abs()) {
        let expected = 1.0 / f.sqrt();
        let actual = <f64 as Float>::inv_sqrt(f);

        if expected > 0. && expected.is_finite() {
            let error = (expected - actual).abs() / expected;
            assert!(error < 1e-6, "{expected} != {actual}");
        } else {
            assert!(actual.is_nan());
        }
    }
}
