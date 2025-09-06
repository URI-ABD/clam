//! Utility functions for the crate. Intended for private use, but made public for testing.

use rayon::prelude::*;

mod distance_value;
mod ord_items;
mod sized_heap;

pub use distance_value::{DistanceValue, FloatDistanceValue};
pub use ord_items::{MaxItem, MinItem};
pub use sized_heap::SizedHeap;

/// The golden ratio `φ = (1 + √5) / 2`
#[allow(clippy::excessive_precision)]
pub const PHI_F64: f64 = 1.618_033_988_749_894_848_204_586_834_365_638_118_f64;

/// The golden ratio `φ = (1 + √5) / 2`
#[allow(clippy::excessive_precision)]
pub const PHI_F32: f32 = 1.618_033_988_749_894_848_204_586_834_365_638_118_f32;

/// Estimates the Local Fractal Dimension (LFD) using the distances of items from a center, and the maximum value among those distances.
///
/// This uses the formula `log2(N / n)`, where `N` is `distances.len() + 1` (the total number of items including the center), and `n` is the number of distances
/// that are less than or equal to `radius / 2` plus one (to account for the center).
///
/// If the radius is zero or if there are no items within half the radius, the LFD is, by definition, `1.0`.
#[expect(clippy::cast_precision_loss)]
pub fn lfd_estimate<T>(distances: &[T], radius: T) -> f64
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

/// Parallel version of [`geometric_median`].
pub fn par_geometric_median<Id: Send + Sync, I: Send + Sync, T: DistanceValue + Send + Sync, M: (Fn(&I, &I) -> T) + Send + Sync>(
    items: &[(Id, I)],
    metric: &M,
) -> usize {
    // Find the index of the item with the minimum total distance to all
    // other items.
    par_pairwise_distances(items, metric)
        .into_par_iter()
        .map(|row| row.into_iter().sum::<T>())
        .enumerate()
        .min_by_key(|&(i, v)| MinItem(i, v))
        .map_or_else(|| unreachable!("items must be non-empty"), |(i, _)| i)
}

/// Returns the index of the geometric median of the given items.
///
/// The geometric median is the item that minimizes the sum of distances to all other items in the slice.
///
/// The user must ensure that the items slice is not empty.
pub fn geometric_median<I, Id, T: DistanceValue, M: Fn(&I, &I) -> T>(items: &[(Id, I)], metric: &M) -> usize {
    // Find the index of the item with the minimum total distance to all other items.
    pairwise_distances(items, metric)
        .into_iter()
        .map(|row| row.into_iter().sum::<T>())
        .enumerate()
        .min_by_key(|&(i, v)| MinItem(i, v))
        .map_or_else(|| unreachable!("items must be non-empty"), |(i, _)| i)
}

/// Parallel version of [`pairwise_distances`].
fn par_pairwise_distances<Id: Send + Sync, I: Send + Sync, T: DistanceValue + Send + Sync, M: (Fn(&I, &I) -> T) + Send + Sync>(
    items: &[(Id, I)],
    metric: &M,
) -> Vec<Vec<T>> {
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
