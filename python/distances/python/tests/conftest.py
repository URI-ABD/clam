"""Pytest configuration file."""

import numpy
import pytest

CAR = 20
MAX_DIM = 4
PARAMS = [10**i for i in range(1, MAX_DIM + 1)]
IDS = [f"10^{i}" for i in range(1, MAX_DIM + 1)]


def gen_data(car: int, dim: int) -> numpy.ndarray:
    """Generate random data."""
    rng = numpy.random.default_rng()
    return rng.random((car, dim))


@pytest.fixture(params=PARAMS, ids=IDS)
def data_f32(request: pytest.FixtureRequest) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    dim: int = request.param
    return gen_data(CAR, dim).astype(numpy.float32)


@pytest.fixture(params=PARAMS, ids=IDS)
def data_f64(request: pytest.FixtureRequest) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    dim: int = request.param
    return gen_data(CAR, dim)


@pytest.fixture(params=PARAMS, ids=IDS)
def data_i32(request: pytest.FixtureRequest) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    dim: int = request.param
    data = (gen_data(CAR, dim) > 0.5).astype(numpy.int32)
    data[data == 0] = -1
    return data


@pytest.fixture(params=PARAMS, ids=IDS)
def data_i64(request: pytest.FixtureRequest) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    dim: int = request.param
    data = (gen_data(CAR, dim) > 0.5).astype(numpy.int64)
    data[data == 0] = -1
    return data


@pytest.fixture(params=PARAMS, ids=IDS)
def data_u32(request: pytest.FixtureRequest) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    dim: int = request.param
    return (gen_data(CAR, dim) > 0.5).astype(numpy.uint32)


@pytest.fixture(params=PARAMS, ids=IDS)
def data_u64(request: pytest.FixtureRequest) -> numpy.ndarray:
    """Return a random array of the given dimension."""
    dim: int = request.param
    return (gen_data(CAR, dim) > 0.5).astype(numpy.uint64)


@pytest.fixture(params=PARAMS[:-2], ids=IDS[:-2])
def strings(request: pytest.FixtureRequest) -> list[str]:
    """Return a list of strings."""
    alphabet = "ACTGN"
    length: int = request.param
    rng = numpy.random.default_rng()
    return ["".join(rng.choice(list(alphabet), length)) for _ in range(CAR)]
