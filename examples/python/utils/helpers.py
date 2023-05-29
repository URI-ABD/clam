import functools
import time
import typing


class TimeIt:
    """ A class to provide a decorator for timing the execution of a function.
    """
    def __init__(self, logger, template: str = 'completed {:s} in {:.3f} seconds'):
        self.template: str = template
        self.logger = logger

    def __call__(self, function: typing.Callable):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):

            start = time.perf_counter()
            result = function(*args, **kwargs)
            end = time.perf_counter()

            self.logger.info(self.template.format(function.__name__, end - start))
            return result

        return wrapper
