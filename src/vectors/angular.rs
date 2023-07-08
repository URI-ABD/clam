//! Angular distances between vectors.

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

    if xx <= U::epsilon() || yy <= U::epsilon() || xy <= U::epsilon() {
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
    U::from(
        x.iter()
            .zip(y.iter())
            .map(|(&a, &b)| a.abs_diff(b) / (a.abs() + b.abs()))
            .sum::<T>(),
    )
}
