//! Utility functions for the crate.

use core::{
    cmp::Ordering,
    f64::{consts::SQRT_2, EPSILON},
};

use distances::Number;

/// Return the index and value of the minimum value in the given slice of values.
///
/// NAN values are ordered as greater than all other values.
///
/// This will return `None` if the given slice is empty.
#[allow(dead_code)]
pub fn arg_min<T: PartialOrd + Copy>(values: &[T]) -> Option<(usize, T)> {
    values
        .iter()
        .enumerate()
        .min_by(|&(_, l), &(_, r)| l.partial_cmp(r).unwrap_or(Ordering::Greater))
        .map(|(i, v)| (i, *v))
}

/// Return the index and value of the maximum value in the given slice of values.
///
/// NAN values are ordered as smaller than all other values.
///
/// This will return `None` if the given slice is empty.
pub fn arg_max<T: PartialOrd + Copy>(values: &[T]) -> Option<(usize, T)> {
    values
        .iter()
        .enumerate()
        .max_by(|&(_, l), &(_, r)| l.partial_cmp(r).unwrap_or(Ordering::Less))
        .map(|(i, v)| (i, *v))
}

/// Return the mean value of the given slice of values.
#[allow(dead_code)]
pub fn mean<T: Number>(values: &[T]) -> f64 {
    values.iter().copied().sum::<T>().as_f64() / values.len().as_f64()
}

/// Return the standard deviation value of the given slice of values.
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// Compute the local fractal dimension of the given distances using the given radius.
///
/// The local fractal dimension is computed as the log2 of the ratio of the number of
/// distances less than or equal to half the radius to the total number of distances.
///
/// # Arguments
///
/// * `radius` - The radius used to compute the distances.
/// * `distances` - The distances to compute the local fractal dimension of.
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

/// Compute the next exponential moving average of the given ratio and parent EMA.
///
/// The EMA is computed as `alpha * ratio + (1 - alpha) * parent_ema`, where `alpha`
/// is a constant value of `2 / 11`. This value was chosen because it gave the best
/// experimental results in the CHAODA paper.
///
/// # Arguments
///
/// * `ratio` - The ratio to compute the EMA of.
/// * `parent_ema` - The parent EMA to use.
#[allow(dead_code)]
pub fn next_ema(ratio: f64, parent_ema: f64) -> f64 {
    // TODO: Consider getting `alpha` from user. Perhaps via env vars?
    let alpha = 2. / 11.;
    alpha.mul_add(ratio, (1. - alpha) * parent_ema)
}

/// Return the position and value of the given value in the given slice of values.
pub fn pos_val<T: Eq + Copy>(values: &[T], v: T) -> Option<(usize, T)> {
    values.iter().copied().enumerate().find(|&(_, x)| x == v)
}
