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

/// Calculate the mean value and standard deviation value of the given slice of values.
///
/// Calculates the mean and standard devitation of the given values using one pass,
/// rather than requiring two passes. This is done by using the following formulas:
///
/// `sum` := `SUM(values)`
/// `sum_squares` := `SUM(values^2)`
/// `mean` := `sum / n`
/// `variance` := `(sum_squares/n) - (sum^2)/(n^2)`
///
/// # Arguments:
///
/// * `values` - The values to calculate the mean and standard deviation of.
///
/// # Returns:
///
/// A tuple containing the mean and standard deviation of the given values.
#[allow(dead_code)]
pub fn mean_variance(values: &[f64]) -> (f64, f64) {
    let n = values.len().as_f64();
    let (sum, sum_squares): (f64, f64) = values
        .iter()
        .fold((0., 0.), |(sum, sum_squares), x| (sum + x, sum_squares + x.powi(2)));

    let mean = sum / n;
    let variance = (sum_squares / n) - (sum.powi(2) / n.powi(2));

    (mean, variance)
}

/// Return the mean value of the given slice of values.
#[allow(dead_code)]
pub fn mean<T: Number>(values: &[T]) -> f64 {
    values.iter().copied().sum::<T>().as_f64() / values.len().as_f64()
}

/// Return the standard deviation value of the given slice of values.
#[allow(dead_code)]
pub fn sd<T: Number>(values: &[T], mean: f64) -> f64 {
    let var = values
        .iter()
        .map(|v| v.as_f64())
        .map(|v| v - mean)
        .map(|v| v.powi(2))
        .sum::<f64>()
        / values.len().as_f64();
    var.sqrt()
}

/// Apply Gaussian normalization to the given values.
#[allow(dead_code)]
pub fn normalize_1d(values: &[f64], mean: f64, sd: f64) -> Vec<f64> {
    values
        .iter()
        .map(|&v| v - mean)
        .map(|v| v / ((EPSILON + sd) * SQRT_2))
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
/// Note that all arrays in the input Vec must have 6 columns.
pub fn rows_to_cols(values: &[[f64; 6]]) -> [Vec<f64>; 6] {
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
///
pub fn calc_row_means(values: &[Vec<f64>; 6]) -> [f64; 6] {
    values
        .iter()
        .map(|values| mean(values))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap_or_else(|_| unreachable!("Array always has a length of 6."))
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
///
pub fn calc_row_sds(values: &[Vec<f64>; 6]) -> [f64; 6] {
    values
        .iter()
        .map(|values| mean_variance(values).1.sqrt())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap_or_else(|_| unreachable!("Array always has a length of 6."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use symagen::random_data;

    #[test]
    fn test_transpose() {
        // Input data: 3 rows x 6 columns
        let data: Vec<[f64; 6]> = vec![
            [2.0, 3.0, 5.0, 7.0, 11.0, 13.0],
            [4.0, 3.0, 5.0, 9.0, 10.0, 15.0],
            [6.0, 2.0, 8.0, 11.0, 9.0, 11.0],
        ];

        // Expected transposed data: 6 rows x 3 columns
        let expected_transposed: [Vec<f64>; 6] = [
            vec![2.0, 4.0, 6.0],
            vec![3.0, 3.0, 2.0],
            vec![5.0, 5.0, 8.0],
            vec![7.0, 9.0, 11.0],
            vec![11.0, 10.0, 9.0],
            vec![13.0, 15.0, 11.0],
        ];

        let transposed_data = rows_to_cols(&data);

        // Check if the transposed data matches the expected result
        for i in 0..6 {
            assert_eq!(transposed_data[i], expected_transposed[i]);
        }
    }

    #[test]
    fn test_means() {
        let all_ratios: Vec<[f64; 6]> = vec![
            [2.0, 4.0, 5.0, 6.0, 9.0, 15.0],
            [3.0, 3.0, 6.0, 4.0, 7.0, 10.0],
            [5.0, 5.0, 8.0, 8.0, 8.0, 1.0],
        ];

        let transposed = rows_to_cols(&all_ratios);
        let means = calc_row_means(&transposed);

        let expected_means: [f64; 6] = [3.3333333333333335, 4.0, 6.333333333333334, 6.0, 8.0, 8.666666666666668];

        means
            .iter()
            .zip(expected_means.iter())
            .for_each(|(&a, &b)| assert!(float_cmp::approx_eq!(f64, a, b, ulps = 2), "{}, {} not equal", a, b));
    }

    #[test]
    fn test_sds() {
        let all_ratios: Vec<[f64; 6]> = vec![
            [2.0, 4.0, 5.0, 6.0, 9.0, 15.0],
            [3.0, 3.0, 6.0, 4.0, 7.0, 10.0],
            [5.0, 5.0, 8.0, 8.0, 8.0, 1.0],
        ];

        let expected_standard_deviations: [f64; 6] = [
            1.2472191289246,
            0.81649658092773,
            1.2472191289246,
            1.6329931618555,
            0.81649658092773,
            5.7927157323276,
        ];
        let sds = calc_row_sds(&rows_to_cols(&all_ratios));

        sds.iter()
            .zip(expected_standard_deviations.iter())
            .for_each(|(&a, &b)| {
                assert!(
                    float_cmp::approx_eq!(f64, a, b, epsilon = 0.00000003),
                    "{}, {} not equal",
                    a,
                    b
                )
            });
    }

    #[test]
    fn test_mean_variance() {
        // Some synthetic cases to test edge results
        let mut test_cases: Vec<Vec<f64>> = vec![
            vec![0.0],
            vec![0.0, 0.0],
            vec![1.0],
            vec![1.0, 2.0],
            vec![0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25],
        ];

        // Use cardinalities of 1, 2, 1000, 100_000 and then 1_000_000 - 10_000_000 in steps of 1_000_000
        let cardinalities = vec![1, 2, 1_000, 100_000]
            .into_iter()
            .chain((1..=10).map(|i| i * 1_000_000))
            .collect::<Vec<_>>();

        // Ranges for the values generated by SyMaGen
        let ranges = vec![
            (-100_000., 0.),
            (-10_000., 0.),
            (-1_000., 0.),
            (0., 1_000.),
            (0., 10_000.),
            (0., 100_000.),
            // These ranges cause the test to fail due to floating point accuracy issues when the sign switches
            //(-1_000., 1_000.),
            //(-10_000., 10_000.),
            //(-100_000., 100_000.)
        ];

        let dimensionality = 1;
        let seed = 42;

        // Generate random data for each cardinality and min/max value where max_val > min_val
        for (cardinality, (min_val, max_val)) in cardinalities.into_iter().zip(ranges.into_iter()) {
            let data = random_data::random_tabular_seedable::<f64>(dimensionality, cardinality, min_val, max_val, seed)
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            test_cases.push(data);
        }

        let (actual_means, actual_variances): (Vec<f64>, Vec<f64>) =
            test_cases.iter().map(|values| mean_variance(values)).unzip();

        // Calculate expected_means and expected_variances using
        // statistical::mean and statistical::population_variance
        let expected_means: Vec<f64> = test_cases.iter().map(|values| statistical::mean(values)).collect();
        let expected_variances: Vec<f64> = test_cases
            .iter()
            .zip(expected_means.iter())
            .map(|(values, &mean)| statistical::population_variance(values, Some(mean)))
            .collect();

        actual_means.iter().zip(expected_means.iter()).for_each(|(&a, &b)| {
            assert!(
                float_cmp::approx_eq!(f64, a, b, ulps = 2),
                "Means not equal. Actual: {}. Expected: {}. Difference: {}.",
                a,
                b,
                a - b
            )
        });

        actual_variances
            .iter()
            .zip(expected_variances.iter())
            .for_each(|(&a, &b)| {
                assert!(
                    float_cmp::approx_eq!(f64, a, b, epsilon = 3e-8),
                    "Variances not equal. Actual: {}. Expected: {}. Difference: {}.",
                    a,
                    b,
                    a - b
                )
            });
    }
}
