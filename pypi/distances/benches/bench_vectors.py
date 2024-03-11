"""Benchmark vector distances."""

from __future__ import annotations

import typing
from functools import partial

import scipy.spatial.distance as scipy_distance
import utils  # type: ignore[import]
from abd_distances import vectors

if typing.TYPE_CHECKING:
    import numpy


CAR = 20
DIM = 500
NUM_RUNS = 100

GEN_DATA = {
    "f32": utils.data_f32,
    "f64": utils.data_f64,
}

METRICS = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "euclidean",
    "sqeuclidean",
    "cityblock",
    "cosine",
]

FUNCTIONS = {
    f"{name}, {dtype}": (
        getattr(scipy_distance, name),
        getattr(vectors, name),
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


PAIR_METRICS = [
    "cdist",
    "pdist",
]

PAIR_FUNCTIONS = {
    f"{name}, {metric}, {dtype}": (
        getattr(scipy_distance, name),
        getattr(vectors, name),
        metric,
        gen_data_x,
        gen_data_x if name == "cdist" else None,
    )
    for name in PAIR_METRICS
    for dtype, gen_data_x in GEN_DATA.items()
    for metric in METRICS
    if "minkowski" not in metric
}


def bench_pair_func(
    func: typing.Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray],
    metric: str,
    gen_data_x: typing.Callable[[int, int], numpy.ndarray],
    gen_data_y: typing.Callable[[int, int], numpy.ndarray] | None,
) -> None:
    """Benchmark a pair distance function."""
    data_x = gen_data_x(CAR * 10, DIM)

    if gen_data_y is None:
        for _ in range(NUM_RUNS):
            _ = func(data_x, metric=metric)  # type: ignore[call-arg]
        return

    data_y = gen_data_y(CAR * 10, DIM)
    for _ in range(NUM_RUNS):
        _ = func(data_x, data_y, metric=metric)  # type: ignore[call-arg]


__benchmarks__ = [
    (
        partial(bench_func, scipy_func, gen_data),
        partial(bench_func, abd_func, gen_data),
        name,
    )
    for name, (scipy_func, abd_func, gen_data) in FUNCTIONS.items()
] + [
    (
        partial(bench_pair_func, scipy_func, metric, gen_data_x, gen_data_y),
        partial(bench_pair_func, abd_func, metric, gen_data_x, gen_data_y),
        name,
    )
    for name, (
        scipy_func,
        abd_func,
        metric,
        gen_data_x,
        gen_data_y,
    ) in PAIR_FUNCTIONS.items()
]
