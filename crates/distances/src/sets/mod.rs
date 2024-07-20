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
        return U::ONE;
    }

    let x = x.iter().copied().collect::<BTreeSet<_>>();
    let y = y.iter().copied().collect::<BTreeSet<_>>();

    let intersection = x.intersection(&y).count();

    if intersection == x.len() && intersection == y.len() {
        U::ZERO
    } else {
        let intersection = U::from(intersection);
        let union = U::from(x.union(&y).count());
        U::ONE - intersection / union
    }
}

/// Dice distance.
///
/// Dice distance, between two sets, measures how dissimilar they are by considering the proportion
/// of elements they don't share in common.  The Dice distance is calculated as twice the ratio of
/// the number of shared elements between two sets to the total number of elements in both sets.
///
/// # Arguments
///
/// * `x`: A set represented as a slice of `Int`s, i.e. a type generic over integers.
/// * `y`: A set represented as a slice of `Int`s, i.e. a type generic over integers.
///
/// # Examples
///
/// ```
/// use distances::sets::dice;
///
/// let x: Vec<u32> = vec![1, 2, 3, 4];
/// let y: Vec<u32> = vec![3, 4, 5, 6];
///
/// let distance: f32 = dice(&x, &y);
///
/// assert!((distance - 0.5).abs() < f32::EPSILON);
/// ```
pub fn dice<T: Int, U: Float>(x: &[T], y: &[T]) -> U {
    if x.is_empty() || y.is_empty() {
        return U::ONE;
    }

    let x = x.iter().copied().collect::<BTreeSet<_>>();
    let y = y.iter().copied().collect::<BTreeSet<_>>();

    let intersection_size = U::from(x.intersection(&y).count());
    let size = U::from(x.len() + y.len());

    if size == U::ZERO {
        return U::ONE;
    }

    U::ONE - ((U::ONE + U::ONE) * intersection_size / size)
}

/// Kulsinski distance.
///
/// Similar to the Jaccard distance, the Kulsinski distance is a measure of the dissimilarity
/// between two sets. It is defined as the sum of the number of not equal dimensions and the
/// total number of dimensions minus the number of elements in the intersection, all divided by
/// the sum of the number of not equal dimensions and the total number of dimensions.
///
/// # Links
///
/// <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html>
/// <https://docs.scipy.org/doc/scipy-1.7.1/reference/reference/generated/scipy.spatial.distance.kulsinski.html>
///
/// # Arguments
///
/// * `x`: A set represented as a slice of `Int`s, i.e. a type generic over integers.
/// * `y`: A set represented as a slice of `Int`s, i.e. a type generic over integers.
///
/// # Examples
///
/// ```
/// use distances::sets::kulsinski;
///
/// let x: Vec<u32> = vec![1, 2, 3];
/// let y: Vec<u32> = vec![2, 3, 4];
///
/// let distance: f32 = kulsinski(&x, &y);
/// let real_distance: f32 = 2_f32 / 3_f32;
///
/// assert!((distance - real_distance).abs() < f32::EPSILON);
/// ```
pub fn kulsinski<T: Int, U: Float>(x: &[T], y: &[T]) -> U {
    if x.is_empty() || y.is_empty() {
        return U::ONE;
    }

    let x = x.iter().copied().collect::<BTreeSet<_>>();
    let y = y.iter().copied().collect::<BTreeSet<_>>();

    let intersection = x.intersection(&y).count();
    let union = x.union(&y).count();
    let not_equal = union - intersection;

    if intersection == x.len() && intersection == y.len() {
        U::ZERO
    } else {
        U::ONE - (U::from(intersection) / U::from(union + not_equal))
    }
}
