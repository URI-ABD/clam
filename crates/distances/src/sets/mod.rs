//! Distance functions for sets.

// use alloc::collections::btree_set::BTreeSet;  // no-std
use std::collections::BTreeSet;

use crate::number::{Float, Int};

/// Jaccard distance.
///
/// The Jaccard distance is a measure of how dissimilar two sets are. It is defined as the
/// cardinality of the intersection of the sets divided by the cardinality of the union of the
/// sets.
///
/// # Arguments
///
/// * `x`: A set represented as a slice of `Int`s, i.e. a type generic over integers.
/// * `y`: A set represented as a slice of `Int`s, i.e. a type generic over integers.
///
/// # Examples
///
/// ```
/// use distances::sets::jaccard;
///
/// let x: Vec<u32> = vec![1, 2, 3];
/// let y: Vec<u32> = vec![2, 3, 4];
///
/// let distance: f32 = jaccard(&x, &y);
///
/// assert!((distance - 0.5).abs() < f32::EPSILON);
/// ```
pub fn jaccard<T: Int, U: Float>(x: &[T], y: &[T]) -> U {
    if x.is_empty() || y.is_empty() {
        return U::one();
    }

    let x = x.iter().copied().collect::<BTreeSet<_>>();
    let y = y.iter().copied().collect::<BTreeSet<_>>();

    let intersection = x.intersection(&y).count();

    if intersection == x.len() && intersection == y.len() {
        U::zero()
    } else {
        let intersection = U::from(intersection);
        let union = U::from(x.union(&y).count());
        U::one() - intersection / union
    }
}
