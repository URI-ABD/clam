//! Utility functions for vector based distance calculations.

use crate::Number;

/// An iterator over the absolute differences between the corresponding elements
/// of two vectors.
pub fn abs_diff_iter<'a, T: Number>(x: &'a [T], y: &'a [T]) -> impl Iterator<Item = T> + 'a {
    x.iter().zip(y.iter()).map(|(&a, &b)| a.abs_diff(b))
}
