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
    let std = (EPSILON + statistical::population_standard_deviation(values, Some(mean))) * SQRT_2;
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
pub fn next_ema(ratio: f64, parent_ema: f64) -> f64 {
    // TODO: Consider getting `alpha` from user. Perhaps via env vars?
    let alpha = 2. / 11.;
    alpha.mul_add(ratio, (1. - alpha) * parent_ema)
}

/// Return the position and value of the given value in the given slice of values.
pub fn pos_val<T: Eq + Copy>(values: &[T], v: T) -> Option<(usize, T)> {
    values.iter().copied().enumerate().find(|&(_, x)| x == v)
}

/// Transpose a matrix represented as an array of arrays (slices) to an array of Vecs.
///
/// Given an array of arrays (slices), where each slice represents a row and each element
/// within the slice represents a column, this function transposes the data to an array of Vecs.
/// The resulting array of Vecs represents the columns of the original matrix. It is expected that each array
/// in the input data has 6 columns.
///
/// # Arguments
///
/// - `all_ratios`: A reference to a Vec of arrays where each array has 6 columns.
///
/// # Returns
///
/// An array of Vecs where each Vec represents a column of the original matrix.
/// /// Note that all arrays in the input Vec must have 6 columns.
///
/// # Panics
///
/// This function may panic if the input data does not have consistent array lengths or if
/// the number of columns in the arrays is not 6.
///
pub fn transpose(values: &[[f64; 6]]) -> [Vec<f64>; 6] {
    let all_ratios: Vec<f64> = values.iter().flat_map(|arr| arr.iter().copied()).collect();
    let mut transposed: [Vec<f64>; 6] = Default::default();

    for (s, element) in transposed.iter_mut().enumerate() {
        *element = all_ratios.iter().skip(s).step_by(6).copied().collect();
    }

    transposed
}

/// Calculate the mean of every row in a 2D array represented as an array of Vecs.
///
/// Given an array of Vecs, where each Vec represents a row and contains a series of f64 values,
/// this function computes the mean for each row. It returns an array of means, where each element
/// corresponds to the mean of the respective row.
///
/// # Arguments
///
/// - `values`: A reference to an array of Vecs, where each Vec represents a row.
///
/// # Returns
///
/// An array of means, where each element represents the mean of a row.
/// # Panics
///
/// This function may panic if the input data does not have consistent row lengths or if the
/// number of rows in the array is not 6.
///
pub fn calc_row_means(values: &[Vec<f64>; 6]) -> [f64; 6] {
    let means: [f64; 6] = values
        .iter()
        .map(|values| statistical::mean(values))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap_or_else(|_| unreachable!("Array always has a length of 6."));

    means
}

/// Calculate the standard deviation of every row in a 2D array represented as an array of Vecs.
///
/// Given an array of Vecs, where each Vec represents a row and contains a series of f64 values,
/// this function computes the standard deviation for each row. It returns an array of standard
/// deviations, where each element corresponds to the standard deviation of the respective row.
///
/// # Arguments
///
/// - `values`: A reference to an array of Vecs, where each Vec represents a row.
///
/// # Returns
///
/// An array of standard deviations, where each element represents the standard deviation of a row.
/// # Panics
///
/// This function may panic if the input data does not have consistent row lengths or if the
/// number of rows in the array is not 6.
///
pub fn calc_row_sds(values: &[Vec<f64>; 6]) -> [f64; 6] {
    let means = calc_row_means(values);
    let sds: [f64; 6] = values
        .iter()
        .zip(means.iter())
        .map(|(values, &mean)| f64::EPSILON + statistical::population_standard_deviation(values, Some(mean)))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap_or_else(|_| unreachable!("Array always has a length of 6"));
    sds
}
