"""Plotting dimensionality reduction using UMAP."""

import pathlib

import numpy
from matplotlib import pyplot as plt


def plot(
    inp_path: pathlib.Path,
    labels_path: pathlib.Path,
    ax: list[plt.Axes],
) -> None:
    """Plot dimensionality reduction results using UMAP."""

    # Load the data
    embedding = numpy.load(inp_path)
    labels = numpy.load(labels_path)

    # Plot the UMAP results
    ax[0].scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=1,
        alpha=0.5,
        c=labels,
    )
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].set_title("UMAP X-Y")

    ax[1].scatter(
        embedding[:, 0],
        embedding[:, 2],
        s=1,
        alpha=0.5,
        c=labels,
    )
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Z")
    ax[1].set_title("UMAP X-Z")

    ax[2].scatter(
        embedding[:, 1],
        embedding[:, 2],
        s=1,
        alpha=0.5,
        c=labels,
    )
    ax[2].set_xlabel("Y")
    ax[2].set_ylabel("Z")
    ax[2].set_title("UMAP Y-Z")
