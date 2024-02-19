"""Pytest configuration file."""

import numpy
import scipy.spatial.distance as scipy_distance
import utils  # type: ignore[import]
from abd_distances import simd

DIM = 1_000
NUM_RUNS = 1_000


def scipy_euclidean_f32() -> None:
    """Benchmark the SciPy implementation of the Euclidean distance."""
    data: numpy.ndarray = utils.data_f32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.euclidean(a, b)


def abd_euclidean_f32() -> None:
    """Benchmark the ABD implementation of the Euclidean distance."""
    data: numpy.ndarray = utils.data_f32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        simd.euclidean_f32(a, b)


def scipy_euclidean_f64() -> None:
    """Benchmark the SciPy implementation of the Euclidean distance."""
    data: numpy.ndarray = utils.data_f64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.euclidean(a, b)


def abd_euclidean_f64() -> None:
    """Benchmark the ABD implementation of the Euclidean distance."""
    data: numpy.ndarray = utils.data_f64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        simd.euclidean_f64(a, b)


def scipy_euclidean_sq_f32() -> None:
    """Benchmark the SciPy implementation of the squared Euclidean distance."""
    data: numpy.ndarray = utils.data_f32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.sqeuclidean(a, b)


def abd_euclidean_sq_f32() -> None:
    """Benchmark the ABD implementation of the squared Euclidean distance."""
    data: numpy.ndarray = utils.data_f32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        simd.euclidean_sq_f32(a, b)


def scipy_euclidean_sq_f64() -> None:
    """Benchmark the SciPy implementation of the squared Euclidean distance."""
    data: numpy.ndarray = utils.data_f64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.sqeuclidean(a, b)


def abd_euclidean_sq_f64() -> None:
    """Benchmark the ABD implementation of the squared Euclidean distance."""
    data: numpy.ndarray = utils.data_f64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        simd.euclidean_sq_f64(a, b)


def scipy_cosine_f32() -> None:
    """Benchmark the SciPy implementation of the Cosine distance."""
    data: numpy.ndarray = utils.data_f32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.cosine(a, b)


def abd_cosine_f32() -> None:
    """Benchmark the ABD implementation of the Cosine distance."""
    data: numpy.ndarray = utils.data_f32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        simd.cosine_f32(a, b)


def scipy_cosine_f64() -> None:
    """Benchmark the SciPy implementation of the Cosine distance."""
    data: numpy.ndarray = utils.data_f64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.cosine(a, b)


def abd_cosine_f64() -> None:
    """Benchmark the ABD implementation of the Cosine distance."""
    data: numpy.ndarray = utils.data_f64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        simd.cosine_f64(a, b)


__benchmarks__ = [
    (scipy_euclidean_f32, abd_euclidean_f32, "SIMD Euclidean, f32"),
    (scipy_euclidean_f64, abd_euclidean_f64, "SIMD Euclidean, f64"),
    (scipy_euclidean_sq_f32, abd_euclidean_sq_f32, "SIMD Euclidean squared, f32"),
    (scipy_euclidean_sq_f64, abd_euclidean_sq_f64, "SIMD Euclidean squared, f64"),
    (scipy_cosine_f32, abd_cosine_f32, "SIMD Cosine, f32"),
    (scipy_cosine_f64, abd_cosine_f64, "SIMD Cosine, f64"),
]
