"""Pytest configuration file."""

import numpy
import scipy.spatial.distance as scipy_distance
import utils  # type: ignore[import]
from abd_distances import vectors

DIM = 1_000
NUM_RUNS = 1_000


def scipy_chebyshev_f32() -> None:
    """Benchmark the SciPy implementation of the Chebyshev distance."""
    data: numpy.ndarray = utils.data_f32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.chebyshev(a, b)


def abd_chebyshev_f32() -> None:
    """Benchmark the ABD implementation of the Chebyshev distance."""
    data: numpy.ndarray = utils.data_f32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        vectors.chebyshev_f32(a, b)


def scipy_chebyshev_f64() -> None:
    """Benchmark the SciPy implementation of the Chebyshev distance."""
    data: numpy.ndarray = utils.data_f64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.chebyshev(a, b)


def abd_chebyshev_f64() -> None:
    """Benchmark the ABD implementation of the Chebyshev distance."""
    data: numpy.ndarray = utils.data_f64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        vectors.chebyshev_f64(a, b)


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
        vectors.euclidean_f32(a, b)


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
        vectors.euclidean_f64(a, b)


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
        vectors.euclidean_sq_f32(a, b)


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
        vectors.euclidean_sq_f64(a, b)


def scipy_l3_distance_f32() -> None:
    """Benchmark the SciPy implementation of the L3 distance."""
    data: numpy.ndarray = utils.data_f32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.minkowski(a, b, 3)


def abd_l3_distance_f32() -> None:
    """Benchmark the ABD implementation of the L3 distance."""
    data: numpy.ndarray = utils.data_f32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        vectors.l3_distance_f32(a, b)


def scipy_l3_distance_f64() -> None:
    """Benchmark the SciPy implementation of the L3 distance."""
    data: numpy.ndarray = utils.data_f64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.minkowski(a, b, 3)


def abd_l3_distance_f64() -> None:
    """Benchmark the ABD implementation of the L3 distance."""
    data: numpy.ndarray = utils.data_f64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        vectors.l3_distance_f64(a, b)


def scipy_l4_distance_f32() -> None:
    """Benchmark the SciPy implementation of the L4 distance."""
    data: numpy.ndarray = utils.data_f32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.minkowski(a, b, 4)


def abd_l4_distance_f32() -> None:
    """Benchmark the ABD implementation of the L4 distance."""
    data: numpy.ndarray = utils.data_f32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        vectors.l4_distance_f32(a, b)


def scipy_l4_distance_f64() -> None:
    """Benchmark the SciPy implementation of the L4 distance."""
    data: numpy.ndarray = utils.data_f64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.minkowski(a, b, 4)


def abd_l4_distance_f64() -> None:
    """Benchmark the ABD implementation of the L4 distance."""
    data: numpy.ndarray = utils.data_f64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        vectors.l4_distance_f64(a, b)


def scipy_manhattan_f32() -> None:
    """Benchmark the SciPy implementation of the Manhattan distance."""
    data: numpy.ndarray = utils.data_f32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.cityblock(a, b)


def abd_manhattan_f32() -> None:
    """Benchmark the ABD implementation of the Manhattan distance."""
    data: numpy.ndarray = utils.data_f32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        vectors.manhattan_f32(a, b)


def scipy_manhattan_f64() -> None:
    """Benchmark the SciPy implementation of the Manhattan distance."""
    data: numpy.ndarray = utils.data_f64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.cityblock(a, b)


def abd_manhattan_f64() -> None:
    """Benchmark the ABD implementation of the Manhattan distance."""
    data: numpy.ndarray = utils.data_f64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        vectors.manhattan_f64(a, b)


def scipy_bray_curtis_u32() -> None:
    """Benchmark the SciPy implementation of the Bray-Curtis distance."""
    data: numpy.ndarray = utils.data_u32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.braycurtis(a, b)


def abd_bray_curtis_u32() -> None:
    """Benchmark the ABD implementation of the Bray-Curtis distance."""
    data: numpy.ndarray = utils.data_u32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        vectors.bray_curtis_u32(a, b)


def scipy_bray_curtis_u64() -> None:
    """Benchmark the SciPy implementation of the Bray-Curtis distance."""
    data: numpy.ndarray = utils.data_u64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.braycurtis(a, b)


def abd_bray_curtis_u64() -> None:
    """Benchmark the ABD implementation of the Bray-Curtis distance."""
    data: numpy.ndarray = utils.data_u64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        vectors.bray_curtis_u64(a, b)


def scipy_canberra_f32() -> None:
    """Benchmark the SciPy implementation of the Canberra distance."""
    data: numpy.ndarray = utils.data_f32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.canberra(a, b)


def abd_canberra_f32() -> None:
    """Benchmark the ABD implementation of the Canberra distance."""
    data: numpy.ndarray = utils.data_f32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        vectors.canberra_f32(a, b)


def scipy_canberra_f64() -> None:
    """Benchmark the SciPy implementation of the Canberra distance."""
    data: numpy.ndarray = utils.data_f64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.canberra(a, b)


def abd_canberra_f64() -> None:
    """Benchmark the ABD implementation of the Canberra distance."""
    data: numpy.ndarray = utils.data_f64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        vectors.canberra_f64(a, b)


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
        vectors.cosine_f32(a, b)


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
        vectors.cosine_f64(a, b)


def scipy_hamming_i32() -> None:
    """Benchmark the SciPy implementation of the Hamming distance."""
    data: numpy.ndarray = utils.data_i32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.hamming(a, b)


def abd_hamming_i32() -> None:
    """Benchmark the ABD implementation of the Hamming distance."""
    data: numpy.ndarray = utils.data_i32(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        vectors.hamming_i32(a, b)


def scipy_hamming_i64() -> None:
    """Benchmark the SciPy implementation of the Hamming distance."""
    data: numpy.ndarray = utils.data_i64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        scipy_distance.hamming(a, b)


def abd_hamming_i64() -> None:
    """Benchmark the ABD implementation of the Hamming distance."""
    data: numpy.ndarray = utils.data_i64(DIM)
    a, b = data[0], data[1]

    for _ in range(NUM_RUNS):
        vectors.hamming_i64(a, b)


__benchmarks__ = [
    (scipy_chebyshev_f32, abd_chebyshev_f32, "Chebyshev, f32"),
    (scipy_chebyshev_f64, abd_chebyshev_f64, "Chebyshev, f64"),
    (scipy_euclidean_f32, abd_euclidean_f32, "Euclidean, f32"),
    (scipy_euclidean_f64, abd_euclidean_f64, "Euclidean, f64"),
    (scipy_euclidean_sq_f32, abd_euclidean_sq_f32, "Euclidean squared, f32"),
    (scipy_euclidean_sq_f64, abd_euclidean_sq_f64, "Euclidean squared, f64"),
    (scipy_l3_distance_f32, abd_l3_distance_f32, "L3, f32"),
    (scipy_l3_distance_f64, abd_l3_distance_f64, "L3, f64"),
    (scipy_l4_distance_f32, abd_l4_distance_f32, "L4, f32"),
    (scipy_l4_distance_f64, abd_l4_distance_f64, "L4, f64"),
    (scipy_manhattan_f32, abd_manhattan_f32, "Manhattan, f32"),
    (scipy_manhattan_f64, abd_manhattan_f64, "Manhattan, f64"),
    (scipy_bray_curtis_u32, abd_bray_curtis_u32, "Bray-Curtis, u32"),
    (scipy_bray_curtis_u64, abd_bray_curtis_u64, "Bray-Curtis, u64"),
    (scipy_canberra_f32, abd_canberra_f32, "Canberra, f32"),
    (scipy_canberra_f64, abd_canberra_f64, "Canberra, f64"),
    (scipy_cosine_f32, abd_cosine_f32, "Cosine, f32"),
    (scipy_cosine_f64, abd_cosine_f64, "Cosine, f64"),
    (scipy_hamming_i32, abd_hamming_i32, "Hamming, i32"),
    (scipy_hamming_i64, abd_hamming_i64, "Hamming, i64"),
]
