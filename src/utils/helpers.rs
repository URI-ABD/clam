use std::f64::consts::SQRT_2;
use std::f64::EPSILON;

use crate::core::number::Number;

/// Return the index and value of the minimum value in the given slice of values.
pub fn arg_min<T: PartialOrd + Copy>(values: &[T]) -> (usize, T) {
    let (i, v) = values
        .iter()
        .enumerate()
        .min_by(|&(_, l), &(_, r)| l.partial_cmp(r).unwrap())
        .unwrap();
    (i, *v)
}

/// Return the index and value of the maximum value in the given slice of values.
pub fn arg_max<T: PartialOrd + Copy>(values: &[T]) -> (usize, T) {
    let (i, v) = values
        .iter()
        .enumerate()
        .max_by(|&(_, l), &(_, r)| l.partial_cmp(r).unwrap())
        .unwrap();
    (i, *v)
}

/// Return the mean value of the given slice of values.
pub fn mean<T: Number>(values: &[T]) -> f64 {
    values.iter().copied().sum::<T>().as_f64() / values.len().as_f64()
}

/// Return the standard deviation value of the given slice of values.
pub fn sd<T: Number>(values: &[T], mean: f64) -> f64 {
    values
        .iter()
        .map(|v| v.as_f64())
        .map(|v| v - mean)
        .map(|v| v.powi(2))
        .sum::<f64>()
        .sqrt()
        / values.len().as_f64()
}

/// Apply Gaussian normalization to the given values.
pub fn normalize_1d(values: &[f64]) -> Vec<f64> {
    let mean = mean(values);
    let std = (EPSILON + sd(values, mean)) * SQRT_2;
    values
        .iter()
        .map(|&v| v - mean)
        .map(|v| v / std)
        .map(libm::erf)
        .map(|v| (1. + v) / 2.)
        .collect()
}

pub fn compute_lfd<T: Number>(radius: T, distances: &[T]) -> f64 {
    if radius == T::zero() {
        1.
    } else {
        let r_2 = radius.as_f64() / 2.;
        let half_count = distances.iter().filter(|&&d| d.as_f64() <= r_2).count();
        if half_count > 0 {
            (distances.len().as_f64() / half_count.as_f64()).log2()
        } else {
            1.
        }
    }
}
