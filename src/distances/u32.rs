use std::f32::EPSILON;

#[allow(clippy::type_complexity)]
pub const METRICS: [(&str, fn(&[u32], &[u32]) -> f32); 4] = [
    ("euclidean", euclidean),
    ("euclidean_sq", euclidean_sq),
    ("manhattan", manhattan),
    ("cosine", cosine),
];

#[inline(always)]
pub fn euclidean(x: &[u32], y: &[u32]) -> f32 {
    euclidean_sq(x, y).sqrt()
}

#[inline(always)]
pub fn euclidean_sq(x: &[u32], y: &[u32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a as f32 - b as f32).powi(2))
        .sum()
}

#[inline(always)]
pub fn manhattan(x: &[u32], y: &[u32]) -> f32 {
    x.iter().zip(y.iter()).map(|(&a, &b)| (a as f32 - b as f32).abs()).sum()
}

#[inline(always)]
pub fn cosine(x: &[u32], y: &[u32]) -> f32 {
    let [xx, yy, xy] = x.iter().zip(y.iter()).fold([0.; 3], |[xx, yy, xy], (&a, &b)| {
        [
            xx + (a as f32 * a as f32),
            yy + (b as f32 * b as f32),
            xy + (a as f32 * b as f32),
        ]
    });

    if xx <= EPSILON || yy <= EPSILON || xy <= EPSILON {
        1.
    } else {
        let d: f32 = 1. - xy / (xx * yy).sqrt();
        if d == 0. {
            0.
        } else {
            d
        }
    }
}
