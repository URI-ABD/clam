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

/// Hausdorff distance function
/// Definition: The Hausdorff distance between two sets of points A and B in a metric space is the greatest of all the distances from a point in A to the nearest point in B.
/// Notes:
/// - can write inf and sup using iterators
/// - fold, map, filter, etc, other adapters
/// - only do it on sets of T where T is numeric: we can look at more later on if we need to
/// - distance function should be able to take in any number of dimensions
/// - use a helper function for distance, which will be euclidian distance
/// - inputs will be two sets, where each item is a vector of some length
/// # Arguments
/// * `x`: A set represented as a slice of `Vec<T>`, e.g. a type generic over vectors of integers.
/// * `y`: A set represented as a slice of `Vec<T>`, e.g. a type generic over vectors of integers.
pub fn hausdorff<T, C, F>(a: &[Vec<T>], b: &[Vec<T>], compare_fn: C, distance_fn: F) -> T
where
    T: Clone,
    C: Fn(&[T], &[T]) -> bool,
    F: Fn(&[T], &[T]) -> T,
{
    // make sure the points in both sets match in length (dimensionality)
    if a.iter().any(|x| b.iter().any(|y| x.len() != y.len())) { 
        panic!("Dimensionalities do not match");
    }

    // initiate variable which will store final hausdorff distance value (supremum)
    // let it be the first element of set a
    let mut h = distance_fn(&a[0], &b[0]); // is there a more efficient way to do this?

    // supremum loop: 'a'
    for x in a.iter() {

        // start with the first element of set b as the shortest distance
        let mut shortest = distance_fn(x, &b[0]);

        // infimum loop
        for y in b.iter() {
            let d = distance_fn(x, y);
            if compare_fn(d, shortest) {
                shortest = d;
            }
        }

        if compare_fn(shortest, h) {
            h = shortest;
        }
    }

    // return final hausdorff value
    h
}

// euclidian helper function (one of many distance functions that could be passed to a meta-distance function like hausdorff)
// should we delete this function?
fn euclidean<T: Int, U: Float>(x: &[T], y: &[T]) -> U {
    let mut sum = U::zero();
    for i in 0..x.len() {
        sum += U::from(x[i] - y[i]).powi(2);
    }
    sum.sqrt()
}