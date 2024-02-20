"""SIMD accelerated distance functions."""

from .abd_distances import simd as abd_simd

euclidean_f32 = abd_simd.euclidean_f32
euclidean_f64 = abd_simd.euclidean_f64
sqeuclidean_f32 = abd_simd.euclidean_sq_f32
sqeuclidean_f64 = abd_simd.euclidean_sq_f64
cosine_f32 = abd_simd.cosine_f32
cosine_f64 = abd_simd.cosine_f64
cdist_f32 = abd_simd.cdist_f32
cdist_f64 = abd_simd.cdist_f64
pdist_f32 = abd_simd.pdist_f32
pdist_f64 = abd_simd.pdist_f64


__all__ = [
    "euclidean_f32",
    "euclidean_f64",
    "sqeuclidean_f32",
    "sqeuclidean_f64",
    "cosine_f32",
    "cosine_f64",
    "cdist_f32",
    "cdist_f64",
    "pdist_f32",
    "pdist_f64",
]
