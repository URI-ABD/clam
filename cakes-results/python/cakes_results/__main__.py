"""The plotting CLI."""

import logging
import pathlib

import typer

import cakes_results

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("cakes_results")
logger.setLevel("INFO")

app = typer.Typer()


@app.command()
def main(
    mode: cakes_results.Mode = typer.Option(
        ...,
        "--mode",
        help="The mode to run the application in.",
        case_sensitive=False,
    ),
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inp-dir",
        help="The directory containing the reports or criterion benchmarks.",
        exists=True,
        file_okay=False,
        readable=True,
        resolve_path=True,
    ),
    plots_dir: pathlib.Path = typer.Option(
        ...,
        "--plots-dir",
        help="The directory to save the plots.",
        exists=True,
        file_okay=False,
        writable=True,
        resolve_path=True,
    ),
    throughput: bool = typer.Option(
        False,
        "--throughput",
        help="Print the throughput instead of plotting the elapsed time.",
    ),
) -> None:
    """The main entry point of the application."""
    if throughput:
        mode.throughput(inp_dir, plots_dir)
    else:
        mode.plot(inp_dir, plots_dir)


if __name__ == "__main__":
    app()
