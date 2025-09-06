"""Benchmark cdist and pdist functions."""

from __future__ import annotations

import typing
from functools import partial

import scipy.spatial.distance as scipy_distance
import utils  # type: ignore[import]
from abd_distances import simd

if typing.TYPE_CHECKING:
    import numpy

CAR = 200
DIM = 500
NUM_RUNS = 100

METRICS = ["euclidean", "sqeuclidean", "cosine"]
GEN_DATA = {
    "f32": utils.data_f32,
    "f64": utils.data_f64,
}


def bench_func(
    func: typing.Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray],
    gen_data_x: typing.Callable[[int, int], numpy.ndarray],
    gen_data_y: typing.Callable[[int, int], numpy.ndarray] | None,
) -> None:
    """Benchmark a distance function."""
    data_x = gen_data_x(CAR, DIM)
    if gen_data_y is None:
        for _ in range(NUM_RUNS):
            func(data_x)  # type: ignore[call-arg]
    else:
        data_y = gen_data_y(CAR, DIM)
        for _ in range(NUM_RUNS):
            func(data_x, data_y)


__benchmarks__ = [
    (
        partial(bench_func, partial(scipy_distance.cdist, metric=name), gen_data, gen_data),
        partial(bench_func, partial(simd.cdist, metric=name), gen_data, gen_data),
        f"SIMD, cdist, {name}, {dtype}",
    )
    for dtype, gen_data in GEN_DATA.items()
    for name in METRICS
] + [
    (
        partial(bench_func, partial(scipy_distance.pdist, metric=name), gen_data, None),
        partial(bench_func, partial(simd.pdist, metric=name), gen_data, None),
        f"SIMD, pdist, {name}, {dtype}",
    )
    for dtype, gen_data in GEN_DATA.items()
    for name in METRICS
]
