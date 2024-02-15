"""Experiments with the `abd_distances` package."""

import inspect
import logging

import abd_distances

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("abd-distances")
logger.setLevel(logging.INFO)


def main() -> None:
    """Experiments with the `abd_distances` package."""
    # Use the `inspect` module to print the members of the `abd_distances` module
    list(map(print, inspect.getmembers(abd_distances.vectors)))

    dist = abd_distances.vectors.euclidean_f32([1, 2, 3], [4, 5, 6])
    logger.info(f"Distance: {dist}")


if __name__ == "__main__":
    main()
