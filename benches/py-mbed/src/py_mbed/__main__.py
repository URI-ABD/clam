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
    inp_dir: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        "--inp-dir",
        "-i",
        help="Path to the directory containing the original input files.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    dataset_name: str = typer.Option(  # noqa: B008
        ...,
        "--dataset-name",
        "-d",
        help="Name of the dataset.",
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
    logger.info(f"Input directory: {inp_dir}")
    logger.info(f"Dataset name: {dataset_name}")
    logger.info(f"Output directory: {out_dir}")
    logger.info("")

    out_dir = out_dir / dataset_name
    if not out_dir.exists():
        out_dir.mkdir(parents=False)

    py_mbed.plot_dim_red(inp_dir, dataset_name, out_dir)

    logger.info("Done.")


if __name__ == "__main__":
    app()
