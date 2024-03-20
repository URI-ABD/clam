"""CLI for making plots."""

from . import plot_strings
from . import plot_vectors


def main() -> None:
    """Run the CLI."""
    plot_strings.make_plots()
    plot_vectors.make_plots()


if __name__ == "__main__":
    main()
