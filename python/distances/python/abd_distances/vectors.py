"""Distances and similarity measures for vectors."""

from .abd_distances import vectors as abd_vectors

chebyshev_f32 = abd_vectors.chebyshev_f32
chebyshev_f64 = abd_vectors.chebyshev_f64
euclidean_f32 = abd_vectors.euclidean_f32
euclidean_f64 = abd_vectors.euclidean_f64
sqeuclidean_f32 = abd_vectors.euclidean_sq_f32
sqeuclidean_f64 = abd_vectors.euclidean_sq_f64
l3_distance_f32 = abd_vectors.l3_norm_f32
l3_distance_f64 = abd_vectors.l3_norm_f64
l4_distance_f32 = abd_vectors.l4_norm_f32
l4_distance_f64 = abd_vectors.l4_norm_f64
manhattan_f32 = abd_vectors.manhattan_f32
manhattan_f64 = abd_vectors.manhattan_f64
braycurtis_u32 = abd_vectors.bray_curtis_u32
braycurtis_u64 = abd_vectors.bray_curtis_u64
canberra_f32 = abd_vectors.canberra_f32
canberra_f64 = abd_vectors.canberra_f64
cosine_f32 = abd_vectors.cosine_f32
cosine_f64 = abd_vectors.cosine_f64
hamming_i32 = abd_vectors.hamming_i32
hamming_i64 = abd_vectors.hamming_i64
cdist_f32 = abd_vectors.cdist_f32
cdist_f64 = abd_vectors.cdist_f64
cdist_u32 = abd_vectors.cdist_u32
cdist_u64 = abd_vectors.cdist_u64
cdist_i32 = abd_vectors.cdist_i32
cdist_i64 = abd_vectors.cdist_i64
pdist_f32 = abd_vectors.pdist_f32
pdist_f64 = abd_vectors.pdist_f64
pdist_u32 = abd_vectors.pdist_u32
pdist_u64 = abd_vectors.pdist_u64
pdist_i32 = abd_vectors.pdist_i32
pdist_i64 = abd_vectors.pdist_i64


__all__ = [
    "chebyshev_f32",
    "chebyshev_f64",
    "euclidean_f32",
    "euclidean_f64",
    "sqeuclidean_f32",
    "sqeuclidean_f64",
    "l3_distance_f32",
    "l3_distance_f64",
    "l4_distance_f32",
    "l4_distance_f64",
    "manhattan_f32",
    "manhattan_f64",
    "braycurtis_u32",
    "braycurtis_u64",
    "canberra_f32",
    "canberra_f64",
    "cosine_f32",
    "cosine_f64",
    "hamming_i32",
    "hamming_i64",
    "cdist_f32",
    "cdist_f64",
    "cdist_u32",
    "cdist_u64",
    "cdist_i32",
    "cdist_i64",
    "pdist_f32",
    "pdist_f64",
    "pdist_u32",
    "pdist_u64",
    "pdist_i32",
    "pdist_i64",
]
