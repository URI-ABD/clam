"""Constants used in the CLAM package."""

import logging
import os

LOG_LEVEL = getattr(logging, os.environ.get("CLAM_LOG", "INFO"))
EMA_SMOOTHING = float(os.environ.get("CLAM_EMA_SMOOTHING", "2"))
EMA_PERIOD = float(os.environ.get("CLAM_EMA_PERIOD", "10"))
EMA_ALPHA = EMA_SMOOTHING / (1 + EMA_PERIOD)

SUBSAMPLE_LIMIT = 100
BATCH_SIZE = 10_000
EPSILON = 1e-6


class Unset:
    """This is a hack around type-hinting when a value cannot be set in __init__.

    https://peps.python.org/pep-0661/

    Usage:

    ```python

    class MyClass:
        def __init__(self, *args, **kwargs):
            ...
            self.__value: typing.Union[ValueType, Unset] = UNSET

        def value_setter(self, *args, **kwargs):
            ...
            self.__value = something
            return

        @property
        def value(self) -> ValueType:
            if self.__value is UNSET:
                raise ValueError(
                    f"Please call `value_setter` before using this property."
                )
            return self.__value
    ```
    """

    __unset = None

    def __new__(cls) -> "Unset":
        """Return the singular instance of `Unset`."""
        if cls.__unset is None:
            cls.__unset = super().__new__(cls)
        return cls.__unset


UNSET = Unset()
