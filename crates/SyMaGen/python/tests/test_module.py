"""Tests that the version is correct and that the module is importable."""

import inspect

import symagen


def test_version() -> None:
    assert symagen.__version__ == "0.3.0"


def test_module() -> None:
    modules = [
        name for name, _ in inspect.getmembers(symagen) if not name.startswith("_")
    ]
    assert "guess_the_number" in modules
