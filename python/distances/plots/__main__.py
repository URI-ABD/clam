"""CLI for making plots."""

from . import plots_simd


def main() -> None:
    """Run the CLI."""
    plots_simd.plot_simd_f32()


if __name__ == "__main__":
    main()
