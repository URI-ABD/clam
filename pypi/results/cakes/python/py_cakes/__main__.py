"""CLI for the package."""

import logging
import pathlib

import typer

from py_cakes import tables
from py_cakes.wrangling_logs import wrangle_logs

logger = logging.getLogger("py_cakes")
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


app = typer.Typer()


@app.command()
def main(
    pre_trim_path: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        help="Path to the file to analyze.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    post_trim_path: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        help="Path to the file to analyze.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Main entry point."""
    if "logs" in str(pre_trim_path):
        wrangle_logs(pre_trim_path)

    if "logs" in str(post_trim_path):
        wrangle_logs(post_trim_path)

    tables.draw_plots(pre_trim_path, post_trim_path)


if __name__ == "__main__":
    app()
