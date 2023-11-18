"""CLI command to create the plots for the scaling results of the Cakes search."""

import concurrent.futures
import logging
import pathlib

import tqdm
import typer

from . import create_plots as _create_plots

# Initialize the logger
logger = logging.getLogger("scaling")
logger.setLevel("INFO")

app = typer.Typer()


@app.command()
def create_plots(
    input_dir: pathlib.Path = typer.Option(
        ...,
        "--input-dir",
        "-i",
        help="The directory containing the reports from the scaling experiments.",
        exists=True,
        readable=True,
        file_okay=False,
        resolve_path=True,
    ),
    output_dir: pathlib.Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="The directory to save the plots.",
        exists=True,
        writable=True,
        file_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Create the plots for the scaling results of the Cakes search."""
    logger.info(f"input_dir = {input_dir}")
    logger.info(f"output_dir = {output_dir}")

    files = list(input_dir.glob("*.json"))
    logger.info(f"Found {len(files)} json files.")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures: list[concurrent.futures.Future[bool]] = []
        for f in files:
            futures.append(
                executor.submit(_create_plots, f, False, output_dir),
            )

        for f in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing files",
        ):
            f.result()  # type: ignore[attr-defined]
