//! Data generation utilities for testing.

use abd_clam::FlatVec;
use distances::{number::Float, Number};
use rand::prelude::*;

pub fn gen_tiny_data() -> FlatVec<Vec<i32>, usize> {
    let items = vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8], vec![11, 12]];
    FlatVec::new_array(items).unwrap_or_else(|e| unreachable!("{e}"))
}

pub fn gen_pathological_line() -> FlatVec<f64, usize> {
    let min_delta = 1e-12;
    let mut delta = min_delta;
    let mut line = vec![0_f64];

    while line.len() < 900 {
        let last = *line.last().unwrap_or_else(|| unreachable!());
        line.push(last + delta);
        delta *= 2.0;
        delta += min_delta;
    }

    FlatVec::new(line).unwrap_or_else(|e| unreachable!("{e}"))
}

pub fn gen_line_data(max: i32) -> FlatVec<i32, usize> {
    let data = (-max..=max).collect::<Vec<_>>();
    FlatVec::new(data).unwrap_or_else(|e| unreachable!("{e}"))
}

pub fn gen_grid_data(max: i32) -> FlatVec<(f32, f32), usize> {
    let data = (-max..=max)
        .flat_map(|x| (-max..=max).map(move |y| (x.as_f32(), y.as_f32())))
        .collect::<Vec<_>>();
    FlatVec::new(data).unwrap_or_else(|e| unreachable!("{e}"))
}

pub fn gen_random_data<T: Float>(car: usize, dim: usize, max: T, seed: u64) -> FlatVec<Vec<T>, usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let data = symagen::random_data::random_tabular(car, dim, -max, max, &mut rng);
    FlatVec::new(data).unwrap_or_else(|e| unreachable!("{e}"))
}
