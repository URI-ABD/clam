//! Angular distances between vectors.

use core::cmp::min;

use crate::{
    number::{Float, Int, UInt},
    Number,
};

/// Computes the Cosine distance between two vectors.
///
/// The cosine distance is defined as `1.0 - c` where `c` is the cosine
/// similarity.
///
/// The cosine similarity is defined as the dot product of the two vectors
/// divided by the product of their magnitudes.
///
/// See the [`crate::vectors`] module documentation for information on this
/// function's potentially unexpected behaviors
///
/// # Arguments
///
/// * `x`: A slice of numbers.
/// * `y`: A slice of numbers.
///
/// # Examples
///
/// ```
/// use distances::vectors::cosine;
///
/// let x: Vec<f32> = vec![1.0, 0.0, 0.0];
/// let y: Vec<f32> = vec![0.0, 1.0, 0.0];
///
/// let distance: f32 = cosine(&x, &y);
///
/// assert!((distance - 1.0).abs() < f32::EPSILON);
/// ```
///
/// # References
///
/// * [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
pub fn cosine<T: Number, U: Float>(x: &[T], y: &[T]) -> U {
    let [xx, yy, xy] = x
        .iter()
        .zip(y.iter())
        .fold([T::zero(); 3], |[xx, yy, xy], (&a, &b)| {
            [a.mul_add(a, xx), b.mul_add(b, yy), a.mul_add(b, xy)]
        });
    let [xx, yy, xy] = [U::from(xx), U::from(yy), U::from(xy)];

    if xx < U::epsilon() || yy < U::epsilon() || xy < U::epsilon() {
        U::one()
    } else {
        let d = U::one() - xy * (xx * yy).inv_sqrt();
        if d < U::epsilon() {
            U::zero()
        } else {
            d
        }
    }
}

/// Computes the Hamming distance between two vectors.
///
/// The Hamming distance is defined as the number of positions at which
/// the corresponding elements are different.
///
/// See the [`crate::vectors`] module documentation for information on this
/// function's potentially unexpected behaviors
///
/// # Arguments
///
/// * `x`: A slice of numbers.
/// * `y`: A slice of numbers.
///
/// # Examples
///
/// ```
/// use distances::vectors::hamming;
///
/// let x: Vec<u8> = vec![1, 2, 3];
/// let y: Vec<u8> = vec![1, 2, 3];
///
/// let distance: u8 = hamming(&x, &y);
///
/// assert_eq!(distance, 0);
///
/// let x: Vec<u8> = vec![1, 2, 3];
/// let y: Vec<u8> = vec![1, 2, 4];
///
/// let distance: u8 = hamming(&x, &y);
///
/// assert_eq!(distance, 1);
/// ```
///
/// # References
///
/// * [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance)
pub fn hamming<T: Int, U: UInt>(x: &[T], y: &[T]) -> U {
    U::from(x.iter().zip(y.iter()).filter(|(&a, &b)| a != b).count())
}

/// Computes the Canberra distance between two vectors.
///
/// The Canberra distance is defined as the sum of the absolute differences
/// between the elements of the two vectors divided by the sum of the absolute
/// values of the elements of the two vectors.
///
/// See the [`crate::vectors`] module documentation for information on this
/// function's potentially unexpected behaviors
///
/// # Arguments
///
/// * `x`: A slice of numbers.
/// * `y`: A slice of numbers.
///
/// # Examples
///
/// ```
/// use distances::vectors::canberra;
///
/// let x: Vec<f32> = vec![1.0, 2.0, 3.0];
/// let y: Vec<f32> = vec![4.0, 5.0, 6.0];
///
/// let distance: f32 = canberra(&x, &y);
///
/// assert!((distance - 143.0 / 105.0).abs() <= f32::EPSILON);
/// ```
///
/// # References
///
/// * [Canberra distance](https://en.wikipedia.org/wiki/Canberra_distance)
pub fn canberra<T: Number, U: Float>(x: &[T], y: &[T]) -> U {
    x.iter()
        .map(|&v| U::from(v))
        .zip(y.iter().map(|&v| U::from(v)))
        .map(|(a, b)| a.abs_diff(b) / (a.abs() + b.abs()))
        .fold(U::zero(), |acc, v| acc + v)
}

/// Computes the Bray-Curtis distance between two vectors.
///
/// The Bray-Curtis dissimilarity is typically used in ecology to
/// measure biodiversity between two sites. Each entry in a vector
/// typically represents the number of observations of a particular
/// species. The distance is defined as 1 minus twice the sum of
/// the minimum observations component-wise divided by the sum all the
/// observations.
///
/// # Arguments
///
/// * `x`: A slice of numbers.
/// * `y`: A slice of numbers.
///
///
/// # Examples
/// ```
/// use distances::vectors::bray_curtis;
///
/// let x: Vec<usize>  = vec![6, 7, 4];
/// let y: Vec<usize> = vec![10, 0, 6];
///
/// let distance: f32 =  bray_curtis(&x, &y);
///
/// assert_eq!(distance, 0.39393938);
/// ```
///
/// # References
///
/// * [Bray-Curtis dissimilarity](https://en.wikipedia.org/wiki/Bray%E2%80%93Curtis_dissimilarity)
pub fn bray_curtis<T: UInt, U: Float>(x: &[T], y: &[T]) -> U {
    let [sum_x, sum_y, sum_min] = x
        .iter()
        .zip(y.iter())
        .fold([T::zero(); 3], |[sum_a, sum_b, sum_min], (&a, &b)| {
            [a + sum_a, b + sum_b, min(a, b) + sum_min]
        });

    if sum_x == T::zero() || sum_y == T::zero() || sum_min == T::zero() {
        U::one()
    } else {
        let (numerator, denominator) = (sum_min + sum_min, sum_x + sum_y);
        if denominator <= numerator {
            U::zero()
        } else {
            U::one() - U::from(numerator) / U::from(denominator)
        }
    }
}
