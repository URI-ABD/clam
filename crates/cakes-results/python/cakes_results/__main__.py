"""Provides the CLI for the Image Calculator plugin."""

import logging

import typer

from cakes_results import scaling

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("cakes-results")
logger.setLevel("INFO")

app = typer.Typer()
app.add_typer(scaling.app, name="scaling")


if __name__ == "__main__":
    app()
