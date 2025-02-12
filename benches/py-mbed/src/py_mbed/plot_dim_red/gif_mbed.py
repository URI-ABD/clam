"""Create a GIF of the dimensionality reduction process."""

import pathlib

import imageio
import numpy
from matplotlib import pyplot as plt

from py_mbed import utils

logger = utils.configure_logger(__name__, "INFO")


def plot(
    mbed_dir: pathlib.Path,
    mbed_name: str,
    out_dir: pathlib.Path,
):
    """Create a GIF of the dimensionality reduction process."""
    # The pattern for the files is `mbed_name-step-<step_number>.npy`.
    mbed_files = list(mbed_dir.glob(f"{mbed_name}-step-*.npy"))
    mbed_files.sort(key=lambda f: int(f.stem.split("-")[-1]))

    mbed_stack: numpy.ndarray = numpy.stack([numpy.load(f) for f in mbed_files])
    min_x: float = mbed_stack.min(axis=1, initial=numpy.inf).min()
    max_x: float = mbed_stack.max(axis=1, initial=-numpy.inf).max()
    min_y: float = mbed_stack.min(axis=2, initial=numpy.inf).min()
    max_y: float = mbed_stack.max(axis=2, initial=-numpy.inf).max()

    # Create the GIF.
    frame_path = out_dir / f"{mbed_name}-frame-temp.png"
    frames = []
    for i in range(len(mbed_files)):
        logger.info(f"Creating frame {i + 1}/{len(mbed_files)}...")
        # Save a temporary frame.
        _plot_frame(
            mbed_stack[i],
            min_x,
            max_x,
            min_y,
            max_y,
            frame_path,
        )
        # Read the temporary frame and append it to the GIF.
        frames.append(imageio.imread(frame_path))
        # Clean up the temporary frame.
        frame_path.unlink()

    logger.info("Creating GIF...")
    gif_path = out_dir / f"{mbed_name}-reduction.gif"
    imageio.mimsave(gif_path, frames, fps=10)


def _plot_frame(
    arr: numpy.ndarray,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    out_path: pathlib.Path,
):
    """Plot a frame of the dimensionality reduction process."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        arr[:, 0],
        arr[:, 1],
        s=10,
        alpha=0.5,
    )
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
