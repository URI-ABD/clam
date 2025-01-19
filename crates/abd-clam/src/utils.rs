//! Utility functions for the crate.

use core::cmp::Ordering;

use distances::{number::Float, Number};
use rand::prelude::*;

/// The square root threshold for sub-sampling.
pub(crate) const SQRT_THRESH: usize = 1000;
/// The logarithmic threshold for sub-sampling.
pub(crate) const LOG2_THRESH: usize = 100_000;

/// Reads the `MAX_RECURSION_DEPTH` environment variable to determine the
/// stride for iterative partition and adaptation.
#[must_use]
pub fn max_recursion_depth() -> usize {
    std::env::var("MAX_RECURSION_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128)
}

/// Return the number of samples to take from the given population size so as to
/// achieve linear time complexity for geometric median estimation.
///
/// The number of samples is calculated as follows:
///
/// - The first `sqrt_thresh` samples are taken from the population.
/// - From the next `log2_thresh` samples, `n` samples are taken, where `n`
///   is the square root of the population size minus `sqrt_thresh`.
/// - From the remaining samples, `n` samples are taken, where `n` is the
///   logarithm base 2 of the population size minus (`log2_thresh` plus
///   `sqrt_thresh`).
#[must_use]
pub fn num_samples(population_size: usize, sqrt_thresh: usize, log2_thresh: usize) -> usize {
    if population_size < sqrt_thresh {
        population_size
    } else {
        sqrt_thresh
            + if population_size < sqrt_thresh + log2_thresh {
                (population_size - sqrt_thresh).as_f64().sqrt()
            } else {
                log2_thresh.as_f64().sqrt() + (population_size - sqrt_thresh - log2_thresh).as_f64().log2()
            }
            .as_usize()
    }
}

/// Choose a subset of the given items using the given thresholds.
///
/// See the `num_samples` function for more information on how the number of
/// samples is calculated.
pub fn choose_samples<T: Clone>(indices: &[T], sqrt_thresh: usize, log2_thresh: usize) -> Vec<T> {
    let mut indices = indices.to_vec();
    let n = crate::utils::num_samples(indices.len(), sqrt_thresh, log2_thresh);
    indices.shuffle(&mut rand::thread_rng());
    indices.truncate(n);
    indices
}

/// Returns the number of distinct pairs that can be formed from `n` elements
/// without repetition.
#[must_use]
pub const fn n_pairs(n: usize) -> usize {
    n * (n - 1) / 2
}

/// Return the index and value of the minimum value in the given slice of values.
///
/// NAN values are ordered as greater than all other values.
///
/// This will return `None` if the given slice is empty.
pub fn arg_min<T: PartialOrd + Number>(values: &[T]) -> Option<(usize, T)> {
    values
        .iter()
        .enumerate()
        .min_by(|&(_, l), &(_, r)| l.total_cmp(r))
        .map(|(i, v)| (i, *v))
}

/// Return the index and value of the maximum value in the given slice of values.
///
/// NAN values are ordered as smaller than all other values.
///
/// This will return `None` if the given slice is empty.
pub fn arg_max<T: PartialOrd + Number>(values: &[T]) -> Option<(usize, T)> {
    values
        .iter()
        .enumerate()
        .max_by(|&(_, l), &(_, r)| l.total_cmp(r))
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
        .fold((F::ZERO, F::ZERO), |(sum, sum_squares), (x, xx)| {
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
        .map(|v| v / ((F::EPSILON + sd) * F::SQRT_2))
        .map(F::erf)
        .map(|v| (F::ONE + v) / F::from(2.))
        .collect()
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
    let alpha = F::from(2) / F::from(11);
    alpha.mul_add(ratio, (F::ONE - alpha) * parent_ema)
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
    data.split_first().map(|(&pivot, tail)| {
        let (left, right) = tail.iter().fold((vec![], vec![]), |(mut left, mut right), &next| {
            if next < pivot {
                left.push(next);
            } else {
                right.push(next);
            }
            (left, right)
        });

        (left, pivot, right)
    })
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

/// Un-flattens a vector of data into a vector of vectors.
///
/// # Arguments
///
/// * `data` - The data to un-flatten.
/// * `sizes` - The sizes of the inner vectors.
///
/// # Returns
///
/// A vector of vectors where each inner vector has the size specified in `sizes`.
///
/// # Errors
///
/// * If the number of elements in `data` is not equal to the sum of the elements in `sizes`.
pub fn un_flatten<T>(data: Vec<T>, sizes: &[usize]) -> Result<Vec<Vec<T>>, String> {
    let num_elements: usize = sizes.iter().sum();
    if data.len() != num_elements {
        return Err(format!(
            "Incorrect number of elements. Expected: {num_elements}. Found: {}.",
            data.len()
        ));
    }

    let mut iter = data.into_iter();
    let mut items = Vec::with_capacity(sizes.len());
    for &s in sizes {
        let mut inner = Vec::with_capacity(s);
        for _ in 0..s {
            inner.push(iter.next().ok_or("Not enough elements!")?);
        }
        items.push(inner);
    }
    Ok(items)
}

/// Read a `Number` from a byte slice and increment the offset.
pub fn read_number<T: Number>(bytes: &[u8], offset: &mut usize) -> T {
    let num_bytes = T::NUM_BYTES;
    let value = T::from_le_bytes(
        bytes[*offset..*offset + num_bytes]
            .try_into()
            .unwrap_or_else(|e| unreachable!("{e}")),
    );
    *offset += num_bytes;
    value
}

/// Reads an encoded value from a byte array and increments the offset.
pub fn read_encoding(bytes: &[u8], offset: &mut usize) -> Box<[u8]> {
    let len = read_number::<usize>(bytes, offset);
    let encoding = bytes[*offset..*offset + len].to_vec();
    *offset += len;
    encoding.into_boxed_slice()
}
