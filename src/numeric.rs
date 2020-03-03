extern crate num;
extern crate rayon;

use std::iter::Sum;

use num::Float;

use rayon::prelude::*;

pub trait Numeric: Float + Sum + Send + Sync {}

impl Numeric for f64 {}
impl Numeric for f32 {}

mod linalg {
    use super::*;

    pub fn dot<T: Numeric>(x: &[T], y: &[T]) -> T {
        x.par_iter().zip(y.par_iter()).map(|(&a, &b)| a * b).sum()
    }
}

pub fn euclidean<T: Numeric>(x: &[T], y: &[T]) -> T {
    euclideansq(x, y).sqrt()
}

pub fn euclideansq<T: Numeric>(x: &[T], y: &[T]) -> T {
    x.par_iter()
        .zip(y.par_iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum()
}

pub fn cosine<T: Numeric>(x: &[T], y: &[T]) -> T {
    let num = linalg::dot(x, y);
    let dem = linalg::dot(x, x).sqrt() * linalg::dot(y, y).sqrt();
    num / dem
}

pub fn manhattan<T: Numeric>(x: &[T], y: &[T]) -> T {
    x.par_iter()
        .zip(y.par_iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum()
}
