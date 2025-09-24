//! Data generation utilities for testing.

pub fn tiny() -> Vec<Vec<i32>> {
    vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8], vec![11, 12]]
}

pub fn pathological_line() -> Vec<f64> {
    let min_delta = 1e-12;
    let mut delta = min_delta;
    let mut line = vec![0_f64];

    while line.len() < 900 {
        let last = *line.last().unwrap_or_else(|| unreachable!());
        line.push(last + delta);
        delta *= 2.0;
        delta += min_delta;
    }

    line
}

pub fn line(max: i32) -> Vec<i32> {
    (-max..=max).collect()
}

pub fn grid(max: i32) -> Vec<(f32, f32)> {
    (-max..=max)
        .flat_map(|x| (-max..=max).map(move |y| (x as f32, y as f32)))
        .collect()
}

pub fn tabular(car: usize, dim: usize, min: f32, max: f32) -> Vec<Vec<f32>> {
    let mut rng = rand::rng();
    symagen::random_data::random_tabular(car, dim, min, max, &mut rng)
}
