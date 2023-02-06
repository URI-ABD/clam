use rand::distributions::Distribution;
use rand::SeedableRng;

use crate::prelude::*;

pub fn arg_min<T: PartialOrd + Copy>(values: &[T]) -> (usize, T) {
    values.iter().enumerate().fold(
        (0, values[0]),
        |(i_min, v_min), (i, &v)| {
            if v < v_min {
                (i, v)
            } else {
                (i_min, v_min)
            }
        },
    )
}

pub fn arg_max<T: PartialOrd + Copy>(values: &[T]) -> (usize, T) {
    values.iter().enumerate().fold(
        (0, values[0]),
        |(i_max, v_max), (i, &v)| {
            if v > v_max {
                (i, v)
            } else {
                (i_max, v_max)
            }
        },
    )
}

pub fn mean<T: Number>(values: &[T]) -> f64 {
    values.iter().cloned().sum::<T>().as_f64() / values.len() as f64
}

pub fn sd<T: Number>(values: &[T], mean: f64) -> f64 {
    values
        .iter()
        .map(|v| v.as_f64())
        .map(|v| (v - mean) * (v - mean))
        .sum::<f64>()
        .sqrt()
        / (values.len() as f64)
}

pub fn normalize_1d(values: &[f64]) -> Vec<f64> {
    let mean = mean(values);
    let std = 1e-8 + sd(values, mean);
    values
        .iter()
        .map(|value| (value - mean) / (std * 2_f64.sqrt()))
        .map(libm::erf)
        .map(|value| (1. + value) / 2.)
        .collect()
}

pub fn gen_data(seed: u64, [n_rows, n_cols]: [usize; 2]) -> Vec<Vec<f64>> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let distribution = rand::distributions::Uniform::new_inclusive(-10., 10.);
    (0..n_rows)
        .map(|_| (0..n_cols).map(|_| distribution.sample(&mut rng)).collect())
        .collect()
}

pub fn get_lfd(max: f64, radial_distances: &[f64]) -> f64 {
    let half_max = max / 2.;
    let half_count = radial_distances.iter().filter(|&&d| d <= half_max).count();
    if half_count > 0 {
        (radial_distances.len().as_f64() / half_count.as_f64()).log2()
    } else {
        1.
    }
}
