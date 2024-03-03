"""Distances and similarity measures for vectors."""

from .abd_distances import vectors

chebyshev = vectors.chebyshev
euclidean = vectors.euclidean
sqeuclidean = vectors.sqeuclidean
manhattan = vectors.manhattan
minkowski = vectors.minkowski
cosine = vectors.cosine
cdist = vectors.cdist
pdist = vectors.pdist


__all__ = [
    "chebyshev",
    "euclidean",
    "sqeuclidean",
    "manhattan",
    "minkowski",
    "cosine",
    "cdist",
    "pdist",
]
