//! Utility functions, traits and types used in and with the `abd-clam` crate.

use rand::prelude::*;
use rayon::prelude::*;

pub mod common_metrics;
mod distance_value;
#[macro_use]
mod named_algorithm;
mod ord_items;
mod quality_measure;
mod sized_heap;

pub use distance_value::{DistanceValue, FloatDistanceValue};
pub use named_algorithm::NamedAlgorithm;
pub use quality_measure::{MeasurableQuality, MeasuredQuality, QualityMeasurer};

pub(crate) use ord_items::{MaxItem, MinItem};
pub(crate) use sized_heap::SizedHeap;

/// Generates `s` random indices, without replacement, from the range `0..n` using the provided random number generator `rng`.
pub(crate) fn random_indices_in_range<R: rand::Rng>(n: usize, s: usize, rng: &mut R) -> Vec<usize> {
    let mut indices = (0..n).collect::<Vec<_>>();
    indices.shuffle(rng);
    indices.truncate(s);
    indices
}

/// Estimates the Local Fractal Dimension (LFD) using the distances of items from a center, and the maximum value among those distances.
///
/// This uses the formula `log2(N / n)`, where `N` is `distances.len() + 1` (the total number of items including the center), and `n` is the number of distances
/// that are less than or equal to `radius / 2` plus one (to account for the center).
///
/// If the radius is zero or if there are no items within half the radius, the LFD is, by definition, `1.0`.
#[expect(clippy::cast_precision_loss)]
pub(crate) fn lfd_estimate<T>(distances: &[T], radius: T) -> f64
where
    T: DistanceValue,
{
    let half_radius = radius.half();
    if distances.len() < 2 || half_radius.is_zero() {
        // In all three of the following cases, we define LFD to be 1.0:
        //   - No non-center items (singleton cluster)
        //   - One non-center item (cluster with two items)
        //   - Radius is zero or too small to be represented as a non-zero value
        1.0
    } else {
        // The cluster has at least 2 non-center items, so LFD computation is meaningful.
        let half_count = distances.iter().filter(|&&d| d <= half_radius).count();
        ((distances.len() + 1) as f64 / ((half_count + 1) as f64)).log2()
    }
}

/// Returns the index of the geometric median of the given items.
///
/// The geometric median is the item that minimizes the sum of distances to all other items in the slice.
///
/// The user must ensure that the items slice is not empty.
pub(crate) fn geometric_median<I, Id, T: DistanceValue, M: Fn(&I, &I) -> T>(items: &[(Id, I)], metric: &M) -> usize {
    // Find the index of the item with the minimum total distance to all other items.
    pairwise_distances(items, metric)
        .into_iter()
        .map(|row| row.into_iter().sum::<T>())
        .enumerate()
        .min_by_key(|&(i, v)| MinItem(i, v))
        .map_or_else(|| unreachable!("items must be non-empty"), |(i, _)| i)
}

/// Parallel version of [`geometric_median`].
pub(crate) fn par_geometric_median<Id: Send + Sync, I: Send + Sync, T: DistanceValue + Send + Sync, M: (Fn(&I, &I) -> T) + Send + Sync>(
    items: &[(Id, I)],
    metric: &M,
) -> usize {
    profi::prof!("par_geometric_median");
    // Find the index of the item with the minimum total distance to all
    // other items.
    par_pairwise_distances(items, metric)
        .into_iter()
        .map(|row| row.into_iter().sum::<T>())
        .enumerate()
        .min_by_key(|&(i, v)| MinItem(i, v))
        .map_or_else(|| unreachable!("items must be non-empty"), |(i, _)| i)
}

/// Computes the pairwise distances between items using the given metric function.
fn pairwise_distances<I, Id, T: DistanceValue, M: Fn(&I, &I) -> T>(items: &[(Id, I)], metric: &M) -> Vec<Vec<T>> {
    let mut matrix = vec![vec![T::zero(); items.len()]; items.len()];
    for (r, (_, i)) in items.iter().enumerate() {
        for (c, (_, j)) in items.iter().enumerate().take(r) {
            let d = metric(i, j);
            matrix[r][c] = d;
            matrix[c][r] = d;
        }
    }
    matrix
}

/// Parallel version of [`pairwise_distances`].
fn par_pairwise_distances<Id: Send + Sync, I: Send + Sync, T: DistanceValue + Send + Sync, M: (Fn(&I, &I) -> T) + Send + Sync>(
    items: &[(Id, I)],
    metric: &M,
) -> Vec<Vec<T>> {
    profi::prof!("par_pairwise_distances");
    let matrix = vec![vec![T::zero(); items.len()]; items.len()];
    items.par_iter().enumerate().for_each(|(r, (_, i))| {
        items.par_iter().enumerate().take(r).for_each(|(c, (_, j))| {
            let d = metric(i, j);
            // SAFETY: We have exclusive access to each cell in the matrix
            // because every (r, c) pair is unique.
            #[allow(unsafe_code)]
            unsafe {
                let row_ptr = &mut *matrix.as_ptr().cast_mut().add(r);
                row_ptr[c] = d;

                let col_ptr = &mut *matrix.as_ptr().cast_mut().add(c);
                col_ptr[r] = d;
            }
        });
    });
    matrix
}
