pub mod helpers;

pub type Metric = fn(&[f32], &[f32]) -> f32;

pub const METRICS: &[(&str, Metric)] = &[
    ("euclidean", distances::vectors::euclidean),
    ("euclidean_sq", distances::vectors::euclidean_sq),
    ("manhattan", distances::vectors::manhattan),
    ("cosine", distances::vectors::cosine),
];
