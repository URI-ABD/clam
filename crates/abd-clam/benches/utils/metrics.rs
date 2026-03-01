//! Distance functions for benchmarks

/// Computes the Euclidean distance between two vectors.
pub fn euclidean<I: AsRef<[f32]>>(a: &I, b: &I) -> f32 {
    distances::simd::euclidean_f32(a.as_ref(), b.as_ref())
}

/// Computes the cosine distance between two vectors.
pub fn cosine<I: AsRef<[f32]>>(a: &I, b: &I) -> f32 {
    distances::blas::cosine_f32(a.as_ref(), b.as_ref())
}
