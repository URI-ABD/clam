"""SyMaGen: Synthetic Manifold Generation."""

from .symagen import *  # noqa: F403

__doc__ = symagen.__doc__  # type: ignore[name-defined]  # noqa: F405, A001
if hasattr(symagen, "__all__"):  # type: ignore[name-defined]  # noqa: F405
    __all__ = symagen.__all__  # type: ignore[name-defined]   # noqa: F405

__version__ = "0.3.0"
