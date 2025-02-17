"""Create a GIF of the dimensionality reduction process."""

import pathlib
import re

import imageio
import numpy
from matplotlib import pyplot as plt

from py_mbed import utils

logger = utils.configure_logger(__name__, "INFO")


def plot(
    out_dir: pathlib.Path,
    dataset_name: str,
    labels_path: pathlib.Path,
) -> pathlib.Path:
    """Create a GIF of the dimensionality reduction process."""
    labels = numpy.load(labels_path)

    # The pattern for the files is `<dataset_name>-step-<step_number>.npy`.
    pattern = re.compile(dataset_name + "-step-" + r"[0-9]+.npy")
    logger.info(f"Looking for files with pattern '{pattern}' in '{out_dir}'...")
    mbed_steps = [f for f in out_dir.glob("*.npy") if pattern.match(f.name)]

    # Sort the files by the step number.
    mbed_steps.sort(key=lambda f: int(f.stem.split("-")[-1]))

    mbed_stack: numpy.ndarray = numpy.stack([numpy.load(f) for f in mbed_steps])
    x_vals = mbed_stack[:, :, 0].flatten()
    y_vals = mbed_stack[:, :, 1].flatten()
    z_vals = mbed_stack[:, :, 2].flatten()
    min_x, max_x = x_vals.min(), x_vals.max()
    min_y, max_y = y_vals.min(), y_vals.max()
    min_z, max_z = z_vals.min(), z_vals.max()

    # Create the GIF.
    frame_path = out_dir / f"{dataset_name}-frame-temp.png"
    frames = []
    for i in range(len(mbed_steps)):
        logger.info(f"Creating frame {i + 1}/{len(mbed_steps)}...")
        # Save a temporary frame.
        _plot_frame(
            arr=mbed_stack[i],
            labels=labels,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_z=min_z,
            max_z=max_z,
            out_path=frame_path,
        )
        # Read the temporary frame and append it to the GIF.
        frames.append(imageio.imread(frame_path))
        # Clean up the temporary frame.
        frame_path.unlink()

    logger.info("Creating GIF...")
    gif_path = out_dir / f"{dataset_name}-reduction.gif"
    imageio.mimsave(gif_path, frames, fps=5)

    return mbed_steps[-1]


def _plot_frame(
    *,
    arr: numpy.ndarray,
    labels: numpy.ndarray,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    min_z: float,
    max_z: float,
    out_path: pathlib.Path,
):
    """Plot a frame of the dimensionality reduction process."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].scatter(
        arr[:, 0],
        arr[:, 1],
        s=1,
        alpha=0.5,
        c=labels,
    )
    ax[0].set_xlim(min_x, max_x)
    ax[0].set_ylim(min_y, max_y)
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")

    ax[1].scatter(
        arr[:, 0],
        arr[:, 2],
        s=1,
        alpha=0.5,
        c=labels,
    )
    ax[1].set_xlim(min_x, max_x)
    ax[1].set_ylim(min_z, max_z)
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Z")

    ax[2].scatter(
        arr[:, 1],
        arr[:, 2],
        s=1,
        alpha=0.5,
        c=labels,
    )
    ax[2].set_xlim(min_y, max_y)
    ax[2].set_ylim(min_z, max_z)
    ax[2].set_xlabel("Y")
    ax[2].set_ylabel("Z")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
