//! Data generation utilities for testing.

#![expect(clippy::cast_precision_loss)]

use rand::prelude::*;

pub fn pathological_line() -> Vec<f64> {
    let min_delta = 1e-12;
    let mut delta = min_delta;
    let mut line = vec![0_f64];

    while line.len() < 900 {
        let last = *line.last().unwrap_or_else(|| unreachable!());
        line.push(last + delta);
        delta *= 2.0;
        delta += min_delta;
    }

    line
}

pub fn line(max: i32) -> Vec<i32> {
    (-max..=max).collect()
}

pub fn grid(max: i32) -> Vec<(f32, f32)> {
    (-max..=max).flat_map(|x| (-max..=max).map(move |y| (x as f32, y as f32))).collect()
}

fn vector(dim: usize, min: f32, max: f32, rng: &mut rand::rngs::StdRng) -> Vec<f32> {
    #[expect(clippy::unwrap_used)]
    let distr = rand::distr::Uniform::new_inclusive(min, max).unwrap();
    (0..dim).map(|_| rng.sample(distr)).collect()
}

pub fn tabular(car: usize, dim: usize, min: f32, max: f32) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..car).map(|_| vector(dim, min, max, &mut rng)).collect()
}
