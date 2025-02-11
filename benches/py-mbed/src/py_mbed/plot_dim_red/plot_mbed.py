"""Plotting data from dimensionality reduction with CLAM-MBED."""

import pathlib

import numpy
from matplotlib import pyplot as plt


def plot(
    inp_path: pathlib.Path,
    ax: plt.Axes,
) -> None:
    """Plot dimensionality reduction results using CLAM-MBED."""

    # Read the data
    data = numpy.load(inp_path)

    # Plot the data
    ax.scatter(
        data[:, 0],
        data[:, 1],
        s=10,
        alpha=0.5,
    )
    ax.set_title("CLAM-MBED")
