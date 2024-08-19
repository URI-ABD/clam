"""CLI for the package."""

import logging
import pathlib

import typer

from py_cakes.wrangling_logs import clusters_by_depth
from py_cakes.wrangling_logs import count_clusters
from py_cakes.wrangling_logs import track_progress

logger = logging.getLogger("py_cakes")
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


app = typer.Typer()


@app.command()
def main(
    log_path: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        help="Path to the log file to analyze.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Main entry point."""
    msg = f"Analyzing {log_path}..."
    logger.info(msg)

    clusters = count_clusters(log_path)
    depth_counts = clusters_by_depth(clusters)
    progress = track_progress(depth_counts)

    for depth, ((finished, f_count), (started, s_count)) in sorted(progress.items()):
        if finished == started:
            msg = f"Depth {depth}: Finished {finished}"
            logger.debug(msg)
        else:
            left = started - finished
            count = s_count - f_count
            msg = (
                f"Depth {depth:4d}: Working on {left:2d} cluster(s), "
                f"containing {count:8d} instances."
            )
            logger.info(msg)


if __name__ == "__main__":
    app()
