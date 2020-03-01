use rayon::prelude::*;
use std::iter::Sum;
use num::FromPrimitive;

pub trait Categorical: PartialEq + Sync + Send + Sum + FromPrimitive {}
impl Categorical for i128 {}
impl Categorical for i64 {}
impl Categorical for i32 {}
impl Categorical for i16 {}
impl Categorical for i8 {}

pub fn hamming<T: Categorical>(x: &[T], y: &[T]) -> T {
    FromPrimitive::from_usize(
        x.par_iter()
            .zip(y.par_iter())
            .filter(|(a, b)| a != b)
            .count()
    ).unwrap()
}
