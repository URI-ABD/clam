//! Utility functions for vector based distance calculations.

use crate::Number;

/// An iterator over the absolute differences between the corresponding elements
/// of two vectors. If one vector is longer than the other, only the elements
/// from the start with a matching element in the shorter vector will be
/// included.
pub fn abs_diff_iter<'a, T: Number>(x: &'a [T], y: &'a [T]) -> impl Iterator<Item = T> + 'a {
    x.iter().zip(y.iter()).map(|(a, &b)| a.abs_diff(b))
}

// /// An iterator over the differences between the corresponding elements of two
// /// slices. The elements of the second slice are subtracted from those of the
// /// first. It is the user's responsibility to ensure that there is no overflow.
// pub fn diff_iter<'a, T: Number>(x: &'a [T], y: &'a [T]) -> impl Iterator<Item = T> + 'a {
//     x.iter().zip(y.iter()).map(|(&a, &b)| a - b)
// }
