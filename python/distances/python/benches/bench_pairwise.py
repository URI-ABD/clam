"""Pytest configuration file."""

import typing
from functools import partial

import numpy
import scipy.spatial.distance as scipy_distance
import utils  # type: ignore[import]
from abd_distances import simd

DIM = 1_000
NUM_RUNS = 1_000

FUNCTIONS = {
    "cdist SIMD Euclidean, f32": (
        partial(scipy_distance.cdist, metric="euclidean"),
        partial(simd.cdist_f32, metric="euclidean"),
        utils.data_f32,
        utils.data_f32,
    ),
    "cdist SIMD Euclidean, f64": (
        partial(scipy_distance.cdist, metric="euclidean"),
        partial(simd.cdist_f64, metric="euclidean"),
        utils.data_f64,
        utils.data_f64,
    ),
    "cdist SIMD Euclidean squared, f32": (
        partial(scipy_distance.cdist, metric="sqeuclidean"),
        partial(simd.cdist_f32, metric="euclidean_sq"),
        utils.data_f32,
        utils.data_f32,
    ),
    "cdist SIMD Euclidean squared, f64": (
        partial(scipy_distance.cdist, metric="sqeuclidean"),
        partial(simd.cdist_f64, metric="euclidean_sq"),
        utils.data_f64,
        utils.data_f64,
    ),
    "cdist SIMD Cosine, f32": (
        partial(scipy_distance.cdist, metric="cosine"),
        partial(simd.cdist_f32, metric="cosine"),
        utils.data_f32,
        utils.data_f32,
    ),
    "cdist SIMD Cosine, f64": (
        partial(scipy_distance.cdist, metric="cosine"),
        partial(simd.cdist_f64, metric="cosine"),
        utils.data_f64,
        utils.data_f64,
    ),
    "pdist SIMD Euclidean, f32": (
        partial(scipy_distance.pdist, metric="euclidean"),
        partial(simd.pdist_f32, metric="euclidean"),
        utils.data_f32,
        None,
    ),
    "pdist SIMD Euclidean, f64": (
        partial(scipy_distance.pdist, metric="euclidean"),
        partial(simd.pdist_f64, metric="euclidean"),
        utils.data_f64,
        None,
    ),
    "pdist SIMD Euclidean squared, f32": (
        partial(scipy_distance.pdist, metric="sqeuclidean"),
        partial(simd.pdist_f32, metric="euclidean_sq"),
        utils.data_f32,
        None,
    ),
    "pdist SIMD Euclidean squared, f64": (
        partial(scipy_distance.pdist, metric="sqeuclidean"),
        partial(simd.pdist_f64, metric="euclidean_sq"),
        utils.data_f64,
        None,
    ),
    "pdist SIMD Cosine, f32": (
        partial(scipy_distance.pdist, metric="cosine"),
        partial(simd.pdist_f32, metric="cosine"),
        utils.data_f32,
        None,
    ),
    "pdist SIMD Cosine, f64": (
        partial(scipy_distance.pdist, metric="cosine"),
        partial(simd.pdist_f64, metric="cosine"),
        utils.data_f64,
        None,
    ),
}


def bench_func(
    func: typing.Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray],
    gen_data_x: typing.Callable[[int], numpy.ndarray],
    gen_data_y: typing.Optional[typing.Callable[[int], numpy.ndarray]],
) -> None:
    """Benchmark a distance function."""
    data_x = gen_data_x(DIM)
    if gen_data_y is None:
        for _ in range(NUM_RUNS):
            func(data_x)  # type: ignore[call-arg]
    else:
        data_y = gen_data_y(DIM)
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
