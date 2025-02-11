"""CLI for plotting dimensionality reduction results."""

import logging
import pathlib

import typer

import py_mbed

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = py_mbed.utils.configure_logger("MbedBenchmarks", "INFO")

app = typer.Typer()


@app.command()
def plot_dim_red(
    original_path: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "--original-path",
        "-i",
        help="Path to the npy file containing the original data.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    mbed_path: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "--mbed-path",
        "-m",
        help="Path to the npy file containing the data reduced by CLAM-MBED.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    out_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "--out-dir",
        "-o",
        help="Path to the directory to store the output files.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
) -> None:
    """Plot dimensionality reduction results."""
    logger.info("Plotting dimensionality reduction results...")
    logger.info("")
    logger.info(f"Original data: {original_path}")
    logger.info(f"MBED data: {mbed_path}")
    logger.info(f"Output directory: {out_dir}")
    logger.info("")

    py_mbed.plot_dim_red(original_path, mbed_path, out_dir)

    logger.info("Done.")


if __name__ == "__main__":
    app()
