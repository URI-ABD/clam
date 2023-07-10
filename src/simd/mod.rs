//! Provides simd-accelerated euclidean distance functions for vectors.

use simd_euclidean::Vectorized;

/// Computes the euclidean distance between two vectors.
#[must_use]
pub fn euclidean_f32(a: &[f32], b: &[f32]) -> f32 {
    Vectorized::distance(a, b)
}

/// Computes the euclidean distance between two vectors.
#[must_use]
pub fn euclidean_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    Vectorized::squared_distance(a, b)
}

/// Computes the euclidean distance between two vectors.
#[must_use]
pub fn euclidean_f64(a: &[f64], b: &[f64]) -> f64 {
    Vectorized::distance(a, b)
}

/// Computes the euclidean distance between two vectors.
#[must_use]
pub fn euclidean_sq_f64(a: &[f64], b: &[f64]) -> f64 {
    Vectorized::squared_distance(a, b)
}
