"""Pytest configuration file."""

import numpy


def data_f32(car: int, dim: int) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    rng = numpy.random.default_rng()
    return rng.random((car, dim)).astype(numpy.float32)


def data_f64(car: int, dim: int) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    rng = numpy.random.default_rng()
    return rng.random((car, dim)).astype(numpy.float64)


def data_i32(car: int, dim: int) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    rng = numpy.random.default_rng()
    data = rng.integers(0, 2, (car, dim)).astype(numpy.int32)
    data[data == 0] = -1
    return data


def data_i64(car: int, dim: int) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    rng = numpy.random.default_rng()
    data = rng.integers(0, 2, (car, dim)).astype(numpy.int64)
    data[data == 0] = -1
    return data


def data_u32(car: int, dim: int) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    rng = numpy.random.default_rng()
    return rng.integers(0, 2, (car, dim)).astype(numpy.uint32)


def data_u64(car: int, dim: int) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    rng = numpy.random.default_rng()
    return rng.integers(0, 2, (car, dim)).astype(numpy.uint64)
