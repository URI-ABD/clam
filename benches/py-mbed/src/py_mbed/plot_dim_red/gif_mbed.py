"""Create a GIF of the dimensionality reduction process."""

import pathlib
import re
import concurrent.futures
import shutil

import imageio
import numpy
from matplotlib import pyplot as plt
import tqdm

from py_mbed import utils

logger = utils.configure_logger(__name__, "INFO")


def plot(
    out_dir: pathlib.Path,
    dataset_name: str,
    labels_path: pathlib.Path,
) -> pathlib.Path:
    """Create a GIF of the dimensionality reduction process."""
    labels = numpy.load(labels_path)

    stack_path = out_dir / f"{dataset_name}-mbed-stack.npy"
    mbed_stack: numpy.ndarray
    if stack_path.exists():
        mbed_stack = numpy.load(stack_path)
    else:
        # The pattern for the files is `<dataset_name>-step-<step_number>.npy`.
        pattern = re.compile(dataset_name + "-step-" + r"[0-9]+.npy")
        logger.info(f"Looking for files with pattern '{pattern}' in '{out_dir}'...")
        mbed_steps = [f for f in out_dir.glob("*.npy") if pattern.match(f.name)]
        mbed_steps.sort(key=lambda f: int(f.stem.split("-")[-1]))
        mbed_stack = numpy.stack([numpy.load(f) for f in mbed_steps])
        numpy.save(stack_path, mbed_stack)
        for f in mbed_steps:
            f.unlink()
    n_steps = mbed_stack.shape[0]

    x_vals = mbed_stack[:, :, 0].flatten()
    y_vals = mbed_stack[:, :, 1].flatten()
    z_vals = mbed_stack[:, :, 2].flatten()
    x_lims = x_vals.min(), x_vals.max()
    y_lims = y_vals.min(), y_vals.max()
    z_lims = z_vals.min(), z_vals.max()

    # Create the frames for the GIF.
    frames_dir = out_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir()

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(n_steps):
            frame_path = frames_dir / f"{dataset_name}-frame-{i}.png"
            results.append(
                executor.submit(
                    _plot_frame,
                    arr=mbed_stack[i],
                    labels=labels,
                    x_lims=x_lims,
                    y_lims=y_lims,
                    z_lims=z_lims,
                    out_path=frame_path,
                )
            )

    frames = []
    for result in tqdm.tqdm(concurrent.futures.as_completed(results), total=len(results)):
        path: pathlib.Path = result.result()
        i = int(path.stem.split("-")[-1])
        frames.append((i, imageio.imread(path)))
        path.unlink()

    frames.sort(key=lambda x: x[0])
    frames = [f for _, f in frames]

    logger.info("Creating GIF...")
    gif_path = out_dir / f"{dataset_name}-reduction.gif"
    imageio.mimsave(gif_path, frames, fps=5)

    return out_dir / f"{dataset_name}-reduced.npy"


def _plot_frame(
    *,
    arr: numpy.ndarray,
    labels: numpy.ndarray,
    x_lims: tuple[float, float],
    y_lims: tuple[float, float],
    z_lims: tuple[float, float],
    out_path: pathlib.Path,
) -> pathlib.Path:
    """Plot a frame of the dimensionality reduction process."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    sizes = 1 + 10 * labels
    ax[0].scatter(
        arr[:, 0],
        arr[:, 1],
        s=sizes,
        alpha=0.5,
        c=labels,
    )
    ax[0].set_xlim(*x_lims)
    ax[0].set_ylim(*y_lims)
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")

    ax[1].scatter(
        arr[:, 0],
        arr[:, 2],
        s=sizes,
        alpha=0.5,
        c=labels,
    )
    ax[1].set_xlim(*x_lims)
    ax[1].set_ylim(*z_lims)
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Z")

    ax[2].scatter(
        arr[:, 1],
        arr[:, 2],
        s=sizes,
        alpha=0.5,
        c=labels,
    )
    ax[2].set_xlim(*y_lims)
    ax[2].set_ylim(*z_lims)
    ax[2].set_xlabel("Y")
    ax[2].set_ylabel("Z")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()

    return out_path
