"""Pytest configuration file."""

import numpy
import pytest

MAX_DIM = 4
PARAMS = [10**i for i in range(1, MAX_DIM + 1)]
IDS = [f"10^{i}" for i in range(1, MAX_DIM + 1)]


@pytest.fixture(params=PARAMS, ids=IDS)
def data_f32(request: pytest.FixtureRequest) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    dim: int = request.param
    rng = numpy.random.default_rng()
    return rng.random((2, dim)).astype(numpy.float32)


@pytest.fixture(params=PARAMS, ids=IDS)
def data_f64(request: pytest.FixtureRequest) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    dim: int = request.param
    rng = numpy.random.default_rng()
    return rng.random((2, dim)).astype(numpy.float64)


@pytest.fixture(params=PARAMS, ids=IDS)
def data_i32(request: pytest.FixtureRequest) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    dim: int = request.param
    rng = numpy.random.default_rng()
    data = rng.integers(0, 2, (2, dim)).astype(numpy.int32)
    data[data == 0] = -1
    return data


@pytest.fixture(params=PARAMS, ids=IDS)
def data_i64(request: pytest.FixtureRequest) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    dim: int = request.param
    rng = numpy.random.default_rng()
    data = rng.integers(0, 2, (2, dim)).astype(numpy.int64)
    data[data == 0] = -1
    return data


@pytest.fixture(params=PARAMS, ids=IDS)
def data_u32(request: pytest.FixtureRequest) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    dim: int = request.param
    rng = numpy.random.default_rng()
    return rng.integers(0, 2, (2, dim)).astype(numpy.uint32)


@pytest.fixture(params=PARAMS, ids=IDS)
def data_u64(request: pytest.FixtureRequest) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    dim: int = request.param
    rng = numpy.random.default_rng()
    return rng.integers(0, 2, (2, dim)).astype(numpy.uint64)