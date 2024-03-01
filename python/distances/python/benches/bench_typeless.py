"""Benchmark vector distances."""

import typing
from functools import partial

import numpy
import scipy.spatial.distance as scipy_distance
import utils  # type: ignore[import]
from abd_distances import typeless_vectors as vectors

setattr(scipy_distance, "manhattan", scipy_distance.cityblock)  # noqa: B010
setattr(scipy_distance, "minkowski-3", partial(scipy_distance.minkowski, p=3))
setattr(scipy_distance, "minkowski-4", partial(scipy_distance.minkowski, p=4))
setattr(vectors, "cityblock", vectors.manhattan)  # noqa: B010
setattr(vectors, "minkowski-3", partial(vectors.minkowski, p=3))
setattr(vectors, "minkowski-4", partial(vectors.minkowski, p=4))

CAR = 10
DIM = 1_000
NUM_RUNS = 100

GEN_DATA = {
    "f32": utils.data_f32,
    "f64": utils.data_f64,
}

METRICS = [
    "chebyshev",
    "euclidean",
    "sqeuclidean",
    "cityblock",
    "minkowski-3",
    "minkowski-4",
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
    f"{name}-{metric}, {dtype}": (
        getattr(scipy_distance, name),
        getattr(vectors, name),
        metric,
        gen_data_x,
        gen_data_x if name == "cdist" else None,
    )
    for name in PAIR_METRICS
    for dtype, gen_data_x in GEN_DATA.items()
    for metric in METRICS if "minkowski" not in metric
}

def bench_pair_func(
    func: typing.Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray],
    metric: str,
    gen_data_x: typing.Callable[[int, int], numpy.ndarray],
    gen_data_y: typing.Optional[typing.Callable[[int, int], numpy.ndarray]],
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
    for name, (scipy_func, abd_func, metric, gen_data_x, gen_data_y)
    in PAIR_FUNCTIONS.items()
]
