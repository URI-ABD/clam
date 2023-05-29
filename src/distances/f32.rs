use std::f32::EPSILON;

#[allow(clippy::type_complexity)]
pub const METRICS: [(&str, fn(&[f32], &[f32]) -> f32); 4] = [
    ("euclidean", euclidean),
    ("euclidean_sq", euclidean_sq),
    ("manhattan", manhattan),
    ("cosine", cosine),
];

#[inline(always)]
pub fn euclidean(x: &[f32], y: &[f32]) -> f32 {
    euclidean_sq(x, y).sqrt()
}

#[inline(always)]
pub fn euclidean_sq(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).map(|(&a, &b)| (a - b).powi(2)).sum()
}

#[inline(always)]
pub fn manhattan(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).map(|(&a, &b)| (a - b).abs()).sum()
}

#[inline(always)]
pub fn cosine(x: &[f32], y: &[f32]) -> f32 {
    let [xx, yy, xy] = x
        .iter()
        .zip(y.iter())
        .fold([0.; 3], |[xx, yy, xy], (&a, &b)| [xx + a * a, yy + b * b, xy + a * b]);

    if xx <= EPSILON || yy <= EPSILON || xy <= EPSILON {
        1.
    } else {
        let d = 1. - xy / (xx * yy).sqrt();
        if d < EPSILON {
            0.
        } else {
            d
        }
    }
}
