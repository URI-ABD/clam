"""CLI for making plots."""

from . import plots_simd


def main() -> None:
    """Run the CLI."""
    plots_simd.make_plots()


if __name__ == "__main__":
    main()
