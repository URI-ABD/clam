//! Distance functions using BLAS

use rust_blas::{Axpy, Copy, Dot, Nrm2};

/// Computes the Cosine distance between two vectors using BLAS.
#[must_use]
pub fn cosine_f32(x: &[f32], y: &[f32]) -> f32 {
    let xx = Dot::dot(x, x);
    let yy = Dot::dot(y, y);
    let xy = Dot::dot(x, y);
    1.0 - xy / (xx * yy).sqrt()
}

/// Computes the Cosine distance between two vectors using BLAS.
#[must_use]
pub fn cosine_f64(x: &[f64], y: &[f64]) -> f64 {
    let xx = Dot::dot(x, x);
    let yy = Dot::dot(y, y);
    let xy = Dot::dot(x, y);
    1.0 - xy / (xx * yy).sqrt()
}

/// Computes the Euclidean distance between two vectors using BLAS.
#[must_use]
pub fn euclidean_f32(x: &[f32], y: &[f32]) -> f32 {
    let mut diff = vec![0.0; x.len()];
    Copy::copy(x, &mut diff);
    Axpy::axpy(&-1.0, y, &mut diff);
    Nrm2::nrm2(&diff)
}

/// Computes the squared Euclidean distance between two vectors using BLAS.
#[must_use]
pub fn euclidean_sq_f32(x: &[f32], y: &[f32]) -> f32 {
    euclidean_f32(x, y).powi(2)
}

/// Computes the squared Euclidean distance between two vectors using BLAS.
#[must_use]
pub fn euclidean_sq_f64(x: &[f64], y: &[f64]) -> f64 {
    euclidean_f64(x, y).powi(2)
}

/// Computes the Euclidean distance between two vectors using BLAS.
#[must_use]
pub fn euclidean_f64(x: &[f64], y: &[f64]) -> f64 {
    let mut diff = vec![0.0; x.len()];
    Copy::copy(x, &mut diff);
    Axpy::axpy(&-1.0, y, &mut diff);
    Nrm2::nrm2(&diff)
}

/// Computes the dot product between two vectors using BLAS.
#[must_use]
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    Dot::dot(x, y)
}

/// Computes the dot product between two vectors using BLAS.
#[must_use]
pub fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
    Dot::dot(x, y)
}

/// Computes the L2 norm of a vector using BLAS.
#[must_use]
pub fn norm2_f32(x: &[f32]) -> f32 {
    Nrm2::nrm2(x)
}

/// Computes the L2 norm of a vector using BLAS.
#[must_use]
pub fn norm2_f64(x: &[f64]) -> f64 {
    Nrm2::nrm2(x)
}
