"""Plotting data from dimensionality reduction with CLAM-MBED."""

import pathlib

import numpy
from matplotlib import pyplot as plt


def plot(
    inp_path: pathlib.Path,
    labels_path: pathlib.Path,
    ax: list[plt.Axes],
) -> None:
    """Plot dimensionality reduction results using CLAM-MBED."""

    # Read the data
    data = numpy.load(inp_path)
    if labels_path.exists():
        labels = numpy.load(labels_path)
    else:
        labels = numpy.zeros(data.shape[0], dtype=bool)

    # colors will be blue for 0 and red for 1
    colors = ["red" if label else "blue" for label in labels]

    # Plot the x-y scatter plot
    ax[0].scatter(
        data[:, 0],
        data[:, 1],
        s=10,
        alpha=0.5,
        c=colors,
    )
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].set_title("CLAM-MBED X-Y")

    # Plot the x-z scatter plot
    ax[1].scatter(
        data[:, 0],
        data[:, 2],
        s=10,
        alpha=0.5,
        c=colors,
    )
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Z")
    ax[1].set_title("CLAM-MBED X-Z")

    # Plot the y-z scatter plot
    ax[2].scatter(
        data[:, 1],
        data[:, 2],
        s=10,
        alpha=0.5,
        c=colors,
    )
    ax[2].set_xlabel("Y")
    ax[2].set_ylabel("Z")
    ax[2].set_title("CLAM-MBED Y-Z")
