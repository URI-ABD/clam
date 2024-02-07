//! Tests for random data generation within a ball.

use rand::SeedableRng;
use symagen::random_data::n_ball;

fn l2_norm(data: &[f64]) -> f64 {
    data.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn tiny() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for r in 0..3 {
        let radius = 10.0_f64.powi(r);
        for dim in 2..=10 {
            let data = n_ball(dim, radius, &mut rng);
            assert_eq!(data.len(), dim);
            assert!(l2_norm(&data) <= radius);
        }
    }
}

#[test]
fn large() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for r in 0..3 {
        let radius = 10.0_f64.powi(r);
        for dim in (100..1_000_000).step_by(10_000) {
            let data = n_ball(dim, radius, &mut rng);
            assert_eq!(data.len(), dim);
            assert!(l2_norm(&data) <= radius);
        }
    }
}
