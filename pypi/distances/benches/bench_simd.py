"""Benchmark vector distances."""

import typing
from functools import partial

import numpy
import scipy.spatial.distance as scipy_distance
import utils  # type: ignore[import]
from abd_distances import simd

CAR = 20
DIM = 500
NUM_RUNS = 100

GEN_DATA = {
    "f32": utils.data_f32,
    "f64": utils.data_f64,
}

METRICS = [
    "euclidean",
    "sqeuclidean",
    "cosine",
]

FUNCTIONS = {
    f"{name}, {dtype}": (
        getattr(scipy_distance, name),
        getattr(simd, name),
        gen_data,
    )
    for name in METRICS
    for dtype, gen_data in GEN_DATA.items()
}


def bench_func(
    func: typing.Callable[[numpy.ndarray, numpy.ndarray], float],
    gen_data: typing.Callable[[int, int], numpy.ndarray],
) -> None:
    """Benchmark a distance function."""
    data_x = gen_data(CAR, DIM)
    data_y = gen_data(CAR, DIM)
    for _ in range(NUM_RUNS):
        for a in data_x:
            for b in data_y:
                _ = func(a, b)


__benchmarks__ = [
    (
        partial(bench_func, scipy_func, gen_data),
        partial(bench_func, abd_func, gen_data),
        f"SIMD, {name}",
    )
    for name, (scipy_func, abd_func, gen_data) in FUNCTIONS.items()
]
