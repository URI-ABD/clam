"""Benchmark cdist and pdist functions."""

import typing
from functools import partial

import numpy
import scipy.spatial.distance as scipy_distance
import utils  # type: ignore[import]
from abd_distances import simd

CAR = 100
DIM = 1_000
NUM_RUNS = 100

FUNCTIONS = {
    "cdist Euclidean, f32": (
        partial(scipy_distance.cdist, metric="euclidean"),
        partial(simd.cdist, metric="euclidean"),
        utils.data_f32,
        utils.data_f32,
    ),
    "cdist Euclidean, f64": (
        partial(scipy_distance.cdist, metric="euclidean"),
        partial(simd.cdist, metric="euclidean"),
        utils.data_f64,
        utils.data_f64,
    ),
    "cdist SqEuclidean, f32": (
        partial(scipy_distance.cdist, metric="sqeuclidean"),
        partial(simd.cdist, metric="sqeuclidean"),
        utils.data_f32,
        utils.data_f32,
    ),
    "cdist SqEuclidean, f64": (
        partial(scipy_distance.cdist, metric="sqeuclidean"),
        partial(simd.cdist, metric="sqeuclidean"),
        utils.data_f64,
        utils.data_f64,
    ),
    "cdist Cosine, f32": (
        partial(scipy_distance.cdist, metric="cosine"),
        partial(simd.cdist, metric="cosine"),
        utils.data_f32,
        utils.data_f32,
    ),
    "cdist Cosine, f64": (
        partial(scipy_distance.cdist, metric="cosine"),
        partial(simd.cdist, metric="cosine"),
        utils.data_f64,
        utils.data_f64,
    ),
    "pdist Euclidean, f32": (
        partial(scipy_distance.pdist, metric="euclidean"),
        partial(simd.pdist, metric="euclidean"),
        utils.data_f32,
        None,
    ),
    "pdist Euclidean, f64": (
        partial(scipy_distance.pdist, metric="euclidean"),
        partial(simd.pdist, metric="euclidean"),
        utils.data_f64,
        None,
    ),
    "pdist SqEuclidean, f32": (
        partial(scipy_distance.pdist, metric="sqeuclidean"),
        partial(simd.pdist, metric="sqeuclidean"),
        utils.data_f32,
        None,
    ),
    "pdist SqEuclidean, f64": (
        partial(scipy_distance.pdist, metric="sqeuclidean"),
        partial(simd.pdist, metric="sqeuclidean"),
        utils.data_f64,
        None,
    ),
    "pdist Cosine, f32": (
        partial(scipy_distance.pdist, metric="cosine"),
        partial(simd.pdist, metric="cosine"),
        utils.data_f32,
        None,
    ),
    "pdist Cosine, f64": (
        partial(scipy_distance.pdist, metric="cosine"),
        partial(simd.pdist, metric="cosine"),
        utils.data_f64,
        None,
    ),
}


def bench_func(
    func: typing.Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray],
    gen_data_x: typing.Callable[[int, int], numpy.ndarray],
    gen_data_y: typing.Optional[typing.Callable[[int, int], numpy.ndarray]],
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
        partial(bench_func, scipy_func, gen_data_x, gen_data_y),
        partial(bench_func, abd_func, gen_data_x, gen_data_y),
        name,
    )
    for name, (scipy_func, abd_func, gen_data_x, gen_data_y) in FUNCTIONS.items()
]
