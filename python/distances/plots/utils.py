"""Plotting utilities for the benchmarks."""

import pathlib

import numpy
import seaborn

IMAGES_DIR = pathlib.Path(__file__).resolve().parents[1] / "images"


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


def make_plot(
    x: list[int],
    y_abd: list[float],
    y_scipy: list[float],
    fn_name: str,
    dt_name: str,
) -> None:
    """Create and save a plot."""
    seaborn.set_theme(style="whitegrid", rc={"figure.figsize": (8, 5)})

    # Add lines with dots for markers
    plot = seaborn.lineplot(x=x, y=y_abd, marker="o", label="ABD")
    plot = seaborn.lineplot(x=x, y=y_scipy, marker="o", label="SciPy")

    # Set the axis labels.
    # The x-axis is the dimension and the y-axis is the time in microseconds.
    plot.set(xlabel="Dimension", ylabel="Time per distance call (µs)")

    # Set the title
    plot.set_title(f"{fn_name} ({dt_name})")

    # Set a tight layout to remove the whitespace around the plot
    plot.figure.tight_layout()

    # Save with 300 DPI
    plot.figure.savefig(IMAGES_DIR / f"{fn_name}_{dt_name}.png", dpi=200)
    plot.figure.clf()
