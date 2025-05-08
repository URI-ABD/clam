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
/// # Arguments
/// * `a`: A set represented as a slice of `Vec<T>`, e.g. a type generic over vectors of integers
/// * `b`: A set represented as a slice of `Vec<T>`, e.g a type generic over vectors of integers
/// * `compare_fn`: A function that compares two distances and returns a boolean if the first argument is "larger"
/// * `distance_fn`: A function that calculates the distance between two points
/// * `lowest_dist_val`: The lowest distance value to be used in the function (I'm not sure if this was necessary, but it's a feature now)
/// NOTE: this will fail if one of the sets is empty
pub fn hausdorff<T, U, C, F>(a: &[Vec<T>], b: &[Vec<T>], compare_fn: C, distance_fn: F, lowest_dist_val: U) -> U
where
    T: Clone + std::marker::Copy, // type of elements in the sets
    U: Clone + std::marker::Copy, // type of distance
    C: Fn(U, U) -> bool,          // function to compare two distances
    F: Fn(&[T], &[T]) -> U,       // function to calculate distance between two points
{
    // note: using x and y is kind of not a good idea for variable names: I'll replace them if I think of something better

    // make sure the points in both sets match in length (dimensionality)
    if a.iter().any(|x| b.iter().any(|y| x.len() != y.len())) {
        panic!("Dimensionalities do not match");
    }

    // initiate variable which will store final hausdorff distance value (supremum)
    let mut h: U = lowest_dist_val.clone();

    // supremum loop: iterate through all elements of set a
    for x in a.iter() {
        // start with the first element of set b as the shortest distance
        let mut shortest: U = distance_fn(x, &b[0]);

        // infimum loop: iterate through all elements of set b
        for y in b.iter() {
            let d: U = distance_fn(x, y);
            if compare_fn(d, shortest) {
                shortest = d;
            }
        }

        if compare_fn(h, shortest) {
            // note: i swapped the order of h and shortest
            h = shortest;
        }
    }

    // do the same for set b
    let prev_h = h.clone(); // store the value of h before we start the next loop
    h = lowest_dist_val.clone(); // reset h to the lowest distance value

    // supremum loop: iterate through all elements of set b
    for y in b.iter() {
        // start with the first element of set a as the shortest distance
        let mut shortest: U = distance_fn(y, &a[0]);

        // infimum loop: iterate through all elements of set a
        for x in a.iter() {
            let d: U = distance_fn(y, x);
            if compare_fn(d, shortest) {
                shortest = d;
            }
        }

        if compare_fn(h, shortest) {
            h = shortest;
        }
    }

    // return final hausdorff value between prev_h and h
    if compare_fn(h, prev_h) {
        prev_h
    } else {
        h
    }
}
