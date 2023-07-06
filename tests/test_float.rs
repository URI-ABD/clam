use rand::prelude::*;

use distances::number::Float;

#[test]
fn sqrt_f32() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    for _ in 0..10_000_000 {
        let a = rng.gen::<f32>().abs();

        let expected = a.sqrt();
        let actual = <f32 as Float>::sqrt(a);

        assert!((expected - actual).abs() < 1e-6);
    }
}

#[test]
fn sqrt_f64() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    for _ in 0..10_000_000 {
        let a = rng.gen::<f64>().abs();

        let expected = a.sqrt();
        let actual = <f64 as Float>::sqrt(a);

        assert!((expected - actual).abs() < 1e-6);
    }
}

#[test]
fn cbrt_f32() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    for _ in 0..10_000_000 {
        let a = rng.gen::<f32>();

        let expected = a.cbrt();
        let actual = <f32 as Float>::cbrt(a);

        assert!((expected - actual).abs() < 1e-6);
    }
}

#[test]
fn cbrt_f64() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    for _ in 0..10_000_000 {
        let a = rng.gen::<f64>();

        let expected = a.cbrt();
        let actual = <f64 as Float>::cbrt(a);

        assert!((expected - actual).abs() < 1e-6);
    }
}

#[test]
fn inv_sqrt_f32() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    for _ in 0..10_000_000 {
        let a = rng.gen::<f32>().abs();

        let expected = 1.0 / a.sqrt();
        let actual = <f32 as Float>::inv_sqrt(a);

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
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    for _ in 0..10_000_000 {
        let a = rng.gen::<f64>().abs();

        let expected = 1.0 / a.sqrt();
        let actual = <f64 as Float>::inv_sqrt(a);

        if expected > 0. && expected.is_finite() {
            let error = (expected - actual).abs() / expected;
            assert!(error < 1e-6, "{expected} != {actual}");
        } else {
            assert!(actual.is_nan());
        }
    }
}
