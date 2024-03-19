"""Python bindings for ndarray-chunked crate."""

from .ndarray_chunked import *  # type: ignore[name-defined]

__doc__ = ndarray_chunked.__doc__  # type: ignore[name-defined]  # noqa: A001
if hasattr(ndarray_chunked, "__all__"):  # type: ignore[name-defined]
    __all__ = ndarray_chunked.__all__  # type: ignore[name-defined]  # noqa: PLE0605


__version__ = "0.1.0"
