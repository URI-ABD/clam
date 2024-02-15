"""Distances and similarity measures for vectors."""

from .abd_distances import vectors as abd_vectors

chebyshev_f32 = abd_vectors.chebyshev_f32
chebyshev_f64 = abd_vectors.chebyshev_f64
euclidean_f32 = abd_vectors.euclidean_f32
euclidean_f64 = abd_vectors.euclidean_f64
euclidean_sq_f32 = abd_vectors.euclidean_sq_f32
euclidean_sq_f64 = abd_vectors.euclidean_sq_f64
l3_distance_f32 = abd_vectors.l3_norm_f32
l3_distance_f64 = abd_vectors.l3_norm_f64
l4_distance_f32 = abd_vectors.l4_norm_f32
l4_distance_f64 = abd_vectors.l4_norm_f64
manhattan_f32 = abd_vectors.manhattan_f32
manhattan_f64 = abd_vectors.manhattan_f64
bray_curtis_u32 = abd_vectors.bray_curtis_u32
bray_curtis_u64 = abd_vectors.bray_curtis_u64
canberra_f32 = abd_vectors.canberra_f32
canberra_f64 = abd_vectors.canberra_f64
cosine_f32 = abd_vectors.cosine_f32
cosine_f64 = abd_vectors.cosine_f64
hamming_i32 = abd_vectors.hamming_i32
hamming_i64 = abd_vectors.hamming_i64


__all__ = [
    "chebyshev_f32",
    "chebyshev_f64",
    "euclidean_f32",
    "euclidean_f64",
    "euclidean_sq_f32",
    "euclidean_sq_f64",
    "l3_distance_f32",
    "l3_distance_f64",
    "l4_distance_f32",
    "l4_distance_f64",
    "manhattan_f32",
    "manhattan_f64",
    "bray_curtis_u32",
    "bray_curtis_u64",
    "canberra_f32",
    "canberra_f64",
    "cosine_f32",
    "cosine_f64",
    "hamming_i32",
    "hamming_i64",
]
