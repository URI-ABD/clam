//! Distance functions for running tests

use std::iter::Sum;

use abd_clam::DistanceValue;
use num_traits::Float;

pub fn absolute_difference<T: DistanceValue>(a: &T, b: &T) -> T {
    if a < b { *b - *a } else { *a - *b }
}

pub fn manhattan<I: AsRef<[T]>, T: DistanceValue>(a: &I, b: &I) -> T {
    a.as_ref().iter().zip(b.as_ref().iter()).map(|(x, y)| absolute_difference(x, y)).sum()
}

pub fn hypotenuse<T: DistanceValue, U: Float>(a: &(T, T), b: &(T, T)) -> U {
    let (a1, a2) = a;
    let (b1, b2) = b;
    let height = U::from(absolute_difference(a1, b1)).unwrap_or_else(|| unreachable!("Height must be a finite number"));
    let base = U::from(absolute_difference(a2, b2)).unwrap_or_else(|| unreachable!("Base must be a finite number"));
    (height * height + base * base).sqrt()
}

pub fn euclidean<I: AsRef<[T]>, T: DistanceValue, U: Float + Sum>(a: &I, b: &I) -> U {
    a.as_ref()
        .iter()
        .zip(b.as_ref().iter())
        .map(|(x, y)| absolute_difference(x, y))
        .map(|d| U::from(d).unwrap_or_else(|| unreachable!("Distance must be a finite number")))
        .map(|d| d * d)
        .sum::<U>()
        .sqrt()
}

// pub fn levenshtein<I: AsRef<[u8]>>(a: &I, b: &I) -> usize {
//     abd_clam::sz_lev_builder()(a, b)
// }
