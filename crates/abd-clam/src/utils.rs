//! Utility functions for the crate.

use core::cmp::Ordering;

use distances::{number::Float, Number};

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

/// Calculate the mean and variance of the given values.
///
/// Calculates the mean and standard deviation using a single pass algorithm.
///
/// # Arguments:
///
/// * `values` - The values to calculate the mean and variance of.
///
/// # Returns:
///
/// A tuple containing the mean and variance of the given values.
pub fn mean_variance<T: Number, F: Float>(values: &[T]) -> (F, F) {
    let n = F::from(values.len());
    let (sum, sum_squares) = values
        .iter()
        .map(|&x| F::from(x))
        .map(|x| (x, x.powi(2)))
        .fold((F::zero(), F::zero()), |(sum, sum_squares), (x, xx)| {
            (sum + x, sum_squares + xx)
        });

    let mean = sum / n;
    let variance = (sum_squares / n) - mean.powi(2);

    (mean, variance)
}

/// Return the mean value of the given slice of values.
pub fn mean<T: Number, F: Float>(values: &[T]) -> F {
    F::from(values.iter().copied().sum::<T>()) / F::from(values.len())
}

/// Return the variance of the given slice of values.
pub fn variance<T: Number, F: Float>(values: &[T], mean: F) -> F {
    values
        .iter()
        .map(|v| F::from(*v))
        .map(|v| v - mean)
        .map(|v| v.powi(2))
        .sum::<F>()
        / F::from(values.len())
}

/// Apply Gaussian normalization to the given values.
#[allow(dead_code)]
pub(crate) fn normalize_1d<F: Float>(values: &[F], mean: F, sd: F) -> Vec<F> {
    values
        .iter()
        .map(|&v| v - mean)
        .map(|v| v / ((F::epsilon() + sd) * F::SQRT_2))
        .map(F::erf)
        .map(|v| (F::one() + v) / F::from(2.))
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
pub(crate) fn compute_lfd<T: Number>(radius: T, distances: &[T]) -> f64 {
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
#[must_use]
pub fn next_ema<F: Float>(ratio: F, parent_ema: F) -> F {
    // TODO: Consider getting `alpha` from user. Perhaps via env vars?
    let alpha = F::from(2.) / F::from(11.);
    alpha.mul_add(ratio, (F::one() - alpha) * parent_ema)
}

/// Return the index of the given value in the given slice of values.
pub(crate) fn position_of<T: Eq + Copy>(values: &[T], v: T) -> Option<usize> {
    values
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, x)| x == v)
        .map(|(i, _)| i)
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
#[must_use]
pub fn rows_to_cols<F: Float>(values: &[[F; 6]]) -> [Vec<F>; 6] {
    let all_ratios = values.iter().flat_map(|arr| arr.iter().copied()).collect::<Vec<_>>();
    let mut transposed: [Vec<F>; 6] = Default::default();

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
#[must_use]
pub fn calc_row_means<F: Float>(values: &[Vec<F>; 6]) -> [F; 6] {
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
#[must_use]
pub fn calc_row_sds<F: Float>(values: &[Vec<F>; 6]) -> [F; 6] {
    values
        .iter()
        .map(|values| (variance(values, mean::<_, F>(values))).sqrt())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap_or_else(|_| unreachable!("Array always has a length of 6."))
}

/// A helper function for the median function below.
///
/// This function partitions the given data into three parts:
/// - A slice of all values less than the pivot value.
/// - The pivot value.
/// - A slice of all values greater than the pivot value.
///
/// # Arguments
///
/// * `data` - The data to partition.
fn partition<T: Number>(data: &[T]) -> Option<(Vec<T>, T, Vec<T>)> {
    if data.is_empty() {
        None
    } else {
        let (pivot_slice, tail) = data.split_at(1);
        let pivot = pivot_slice[0];
        let (left, right) = tail.iter().fold((vec![], vec![]), |mut splits, next| {
            {
                let (ref mut left, ref mut right) = &mut splits;
                if next < &pivot {
                    left.push(*next);
                } else {
                    right.push(*next);
                }
            }
            splits
        });

        Some((left, pivot, right))
    }
}

/// A helper function for the median function below.
///
/// This function selects the kth smallest element from the given data.
///
/// # Arguments
///
/// * `data` - The data to select the kth smallest element from.
/// * `k` - The index of the element to select.
fn select<T: Number>(data: &[T], k: usize) -> Option<T> {
    let part = partition(data);

    match part {
        None => None,
        Some((left, pivot, right)) => {
            let pivot_idx = left.len();

            match pivot_idx.cmp(&k) {
                Ordering::Equal => Some(pivot),
                Ordering::Greater => select(&left, k),
                Ordering::Less => select(&right, k - (pivot_idx + 1)),
            }
        }
    }
}

/// Find the median value using the quickselect algorithm.
///
/// If the number of elements is odd, the median is the middle element.
/// If the number of elements is even, the median should be the average of the
/// two middle elements, but this implementation returns the lower of the two
/// middle elements.
///
/// Source: <https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/statistics.html>
///
/// # Arguments
///
/// * `data` - The data to find the median of.
pub fn median<T: Number>(data: &[T]) -> Option<T> {
    let size = data.len();

    match size {
        even if even % 2 == 0 => select(data, (even / 2) - 1),
        // },
        // even if even % 2 == 0 => {
        //     let fst_med = select(data, (even / 2) - 1);
        //     let snd_med = select(data, even / 2);

        //     match (fst_med, snd_med) {
        //         (Some(fst), Some(snd)) => Some((fst + snd) / T::from(2)),
        //         _ => None
        //     }
        // },
        odd => select(data, odd / 2),
    }
}

/// Finds the standard deviation
///
/// A helper function to find the standard deviation from a list of values
///
/// Source: <https://en.wikipedia.org/wiki/Standard_deviation>
///
/// # Arguments
///
/// * `values` - The data to find the STD of.
pub fn standard_deviation<T: Number, F: Float>(values: &[T]) -> F {
    variance(values, mean::<_, F>(values)).sqrt()
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use symagen::random_data;

    use super::*;

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

        let expected_means: [f64; 6] = [
            3.333_333_333_333_333_5,
            4.0,
            6.333_333_333_333_334,
            6.0,
            8.0,
            8.666_666_666_666_668,
        ];

        means
            .iter()
            .zip(expected_means.iter())
            .for_each(|(&a, &b)| assert!(float_cmp::approx_eq!(f64, a, b, ulps = 2), "{a}, {b} not equal"));
    }

    #[test]
    fn test_sds() {
        let all_ratios: Vec<[f64; 6]> = vec![
            [2.0, 4.0, 5.0, 6.0, 9.0, 15.0],
            [3.0, 3.0, 6.0, 4.0, 7.0, 10.0],
            [5.0, 5.0, 8.0, 8.0, 8.0, 1.0],
        ];

        let expected_standard_deviations: [f64; 6] = [
            1.247_219_128_924_6,
            0.816_496_580_927_73,
            1.247_219_128_924_6,
            1.632_993_161_855_5,
            0.816_496_580_927_73,
            5.792_715_732_327_6,
        ];
        let sds = calc_row_sds(&rows_to_cols(&all_ratios));

        sds.iter()
            .zip(expected_standard_deviations.iter())
            .for_each(|(&a, &b)| {
                assert!(
                    float_cmp::approx_eq!(f64, a, b, epsilon = 0.000_000_03),
                    "{a}, {b} not equal"
                );
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
            let data = random_data::random_tabular(
                dimensionality,
                cardinality,
                min_val,
                max_val,
                &mut rand::rngs::StdRng::seed_from_u64(seed),
            )
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
            test_cases.push(data);
        }

        let (actual_means, actual_variances): (Vec<f64>, Vec<f64>) = test_cases
            .iter()
            .map(|values| mean_variance::<f64, f64>(values))
            .unzip();

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
            );
        });

        actual_variances
            .iter()
            .zip(expected_variances.iter())
            .for_each(|(&a, &b)| {
                assert!(
                    float_cmp::approx_eq!(f64, a, b, epsilon = 3e-3),
                    "Variances not equal. Actual: {}. Expected: {}. Difference: {}.",
                    a,
                    b,
                    a - b
                );
            });
    }

    #[test]
    fn test_standard_deviation() {
        let data = [2., 4., 4., 4., 5., 5., 7., 9.];
        let std = standard_deviation::<f32, f32>(&data);
        assert!((std - 2.).abs() < 1e-6);
    }
}
