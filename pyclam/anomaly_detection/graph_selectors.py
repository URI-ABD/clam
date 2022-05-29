import abc
import logging
import typing

import numpy

from . import pretrained_models
from .. import core
from .. import utils
from ..core import criterion
from ..utils import constants

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOG_LEVEL)


class GraphSelector(abc.ABC):
    def __init__(
            self,
            name: str,
            function: typing.Callable[[numpy.array], float],
    ):
        """ This creates a graph selector from a trained meta-ml model.

        Args:
            name: of the scoring function formatted as {ml_model}_{metric_name}_{scorer_name} where:
             - `ml_model` is 'lr' (for linear regression), 'dt' (for decision tree), etc.
             - `metric_name` is the name of the distance metric with which the model was trained.
             - `scorer_name` is the name of te individual algorithm with which this model was trained.
            function: The scoring function extracted from the trained model.
        """
        splits = name.split('_')

        self.full_name = name
        self.ml_name = splits[0]
        self.metric_name = splits[1]
        self.scorer_name = '_'.join(splits[2:])
        self.function = function

        self.selector: typing.Union[criterion.MetaMLSelect, utils.Unset] = constants.UNSET

    def configure(self, criteria: typing.Type[criterion.MetaMLSelect], **kwargs) -> 'GraphSelector':
        self.selector = criteria(self.function, **kwargs)
        return self

    def __call__(self, root: core.Cluster) -> core.Graph:
        if isinstance(self.selector, utils.Unset):
            raise ValueError(f'Need to call `configure` on this selector before creating a graph.')
        logger.info(f'Selecting a graph using {self.full_name} ...')
        return core.Graph(*self.selector(root)).build_edges()


DEFAULT_SELECTORS = [
    GraphSelector(name, function)
    for name, function in pretrained_models.META_MODELS.items()
]

__all__ = [
    'DEFAULT_SELECTORS',
    'GraphSelector'
]
