//! Provides functions for calculating correlations between vectors.

use crate::{number::Float, Number};

/// Pearson distance.
///
/// Returns 1.0 - r, where r is the Pearson Correlation Coefficient.
/// Measures linear correlation between two vectors,
/// where 0 is a perfect positive correlation,
/// 1 is no correlation, and 2 is a perfect negative correlation.
pub fn pearson<T: Number, U: Float>(x: &[T], y: &[T]) -> U {
    // Find means of each vector
    let x_sum = x.iter().fold(T::ZERO, |acc, &i| acc + i);
    let x_mean = U::from(x_sum) / U::from(x.len());
    let y_sum = y.iter().fold(T::ZERO, |acc, &i| acc + i);
    let y_mean = U::from(y_sum) / U::from(y.len());

    // Determine covariances and standard deviations
    let covariance = x.iter().zip(y.iter()).fold(U::ZERO, |acc, (&xi, &yi)| {
        acc + (U::from(xi) - x_mean) * (U::from(yi) - y_mean)
    });

    let std_dev_x = x
        .iter()
        .fold(U::ZERO, |acc, &i| acc + (U::from(i) - x_mean) * (U::from(i) - x_mean))
        .sqrt();

    let std_dev_y = y
        .iter()
        .fold(U::ZERO, |acc, &i| acc + (U::from(i) - y_mean) * (U::from(i) - y_mean))
        .sqrt();

    // 1.0 - Pearson correlation coefficient
    U::ONE - (covariance / (std_dev_x * std_dev_y))
}
