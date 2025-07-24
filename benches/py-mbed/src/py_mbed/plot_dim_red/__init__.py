"""Plotting the results of dimensionality reduction."""

import pathlib

from matplotlib import pyplot as plt

import numpy
import umap

from . import gif_mbed
from . import plot_mbed
from . import plot_umap

from py_mbed import utils

logger = utils.configure_logger(__name__, "INFO")


def plot(
    inp_dir: pathlib.Path,
    dataset_name: str,
    out_dir: pathlib.Path,
) -> None:
    """Plot dimensionality reduction results."""

    original_path = inp_dir / f"{dataset_name}.npy"
    labels_path = inp_dir / f"{dataset_name}_labels.npy"
    umap_path = out_dir / f"{dataset_name}-umap.npy"

    mbed_path = gif_mbed.plot(out_dir, dataset_name, labels_path)

    if not umap_path.exists():
        logger.info("Performing UMAP dimensionality reduction...")

        # Read the input data as a numpy array
        raw_data = numpy.load(original_path)

        # Perform UMAP dimensionality reduction
        reducer = umap.UMAP(
            n_neighbors=15,
            n_components=3,
        )

        if raw_data.shape[0] < 25000:
            embedding = reducer.fit_transform(raw_data)
        else:
            logger.info("Data size is too large. Using the random 25000 samples...")
            random_sample = numpy.random.choice(raw_data.shape[0], 25000, replace=False)
            model: umap.UMAP = reducer.fit(raw_data[random_sample])
            embedding = model.transform(raw_data)

        logger.info("Saving UMAP results...")
        # Save the UMAP results
        numpy.save(umap_path, embedding)  # type: ignore

    logger.info("Reduced data already exists. Creating plots...")

    fig: plt.Figure  # type: ignore
    ax: plt.Axes  # type: ignore
    fig, ax = plt.subplots(2, 3, figsize=(20, 15))

    plot_umap.plot(umap_path, labels_path, ax[0])  # type: ignore
    if mbed_path.exists():
        plot_mbed.plot(mbed_path, labels_path, ax[1])  # type: ignore

    plt.tight_layout()
    plt.savefig(out_dir / "dim_red_results.png", dpi=200)
    plt.close()


__all__ = ["plot"]
