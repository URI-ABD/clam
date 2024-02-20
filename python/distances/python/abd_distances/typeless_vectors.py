"""Distances and similarity measures for vectors."""

import numpy

from .abd_distances import typeless_vectors as abd_vectors


def chebyshev(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Chebyshev distance between two vectors."""
    return abd_vectors.chebyshev(a, b)


def euclidean(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return abd_vectors.euclidean(a, b)


def sqeuclidean(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Squared Euclidean distance between two vectors."""
    return abd_vectors.sqeuclidean(a, b)


def manhattan(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Manhattan distance between two vectors."""
    return abd_vectors.manhattan(a, b)


def cosine(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return abd_vectors.cosine(a, b)


def cdist(a: numpy.ndarray, b: numpy.ndarray, metric: str) -> numpy.ndarray:
    """Compute distance between each pair of the two collections of inputs."""
    return abd_vectors.cdist(a, b, metric)


def pdist(a: numpy.ndarray, metric: str) -> numpy.ndarray:
    """Pairwise distances between observations in n-dimensional space."""
    return abd_vectors.pdist(a, metric)


__all__ = [
    "chebyshev",
    "euclidean",
    "sqeuclidean",
    "manhattan",
    "cosine",
    "cdist",
    "pdist",
]
