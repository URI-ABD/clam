"""Plotting utilities for the benchmarks."""

from __future__ import annotations

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


def gen_strings(car: int, dim: int, alphabet: str) -> list[str]:
    """Return a list of random strings."""
    rng = numpy.random.default_rng()
    return ["".join(rng.choice(list(alphabet), dim)) for _ in range(car)]


def make_plot(  # noqa: PLR0913
    *,
    x: list[int],
    y_abd: list[float],
    fn_name: str,
    dt_name: str,
    y_units: str = "µs",
    y_comp: list[float] | None = None,
    x_label: str = "Dimension",
    y_comp_label: str = "SciPy",
) -> None:
    """Create and save a plot."""
    seaborn.set_theme(style="whitegrid", rc={"figure.figsize": (8, 5)})

    # Add lines with dots for markers
    plot = seaborn.lineplot(x=x, y=y_abd, marker="o", label="ABD")
    if y_comp is not None:
        plot = seaborn.lineplot(x=x, y=y_comp, marker="o", label=y_comp_label)

    # Set the axis labels.
    # The x-axis is the dimension and the y-axis is the time.
    plot.set(xlabel=x_label, ylabel=f"Time per distance call ({y_units})")

    # Set the title
    plot.set_title(f"{fn_name} ({dt_name})")

    # Set a tight layout to remove the whitespace around the plot
    plot.figure.tight_layout()

    # Save with 300 DPI
    plot.figure.savefig(IMAGES_DIR / f"{fn_name}_{dt_name}.png", dpi=200)
    plot.figure.clf()
