import abc
import logging

import numpy
from sklearn import linear_model
from sklearn import tree

from ..core import Manifold
from ..utils import constants

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOG_LEVEL)


class MetaMLModel(abc.ABC):

    def __init__(self, model_class, **kwargs):
        """ Creates a model and initializes it with the given key-word arguments.
        """
        self.model = model_class(**kwargs)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """ Fits the model.
        """
        pass

    @abc.abstractmethod
    def extract_python(self, *args, **kwargs) -> tuple[list[str], str]:
        """ Extracts the scoring function as a string which can be written to disk.

        The scoring function should have the following signature:

        ```
        def helpful_function_name(ratios: numpy.ndarray) -> float:
            ...
        ```

        where `ratios` is a 1d array of floats with 6 elements. These ratios are
        `cluster_ratios` from Manifold.

        Returns:
            A 2-tuple whose items are:
            - a list of strings which will be written as the import lines.
            - a string containing the code for the scoring function in Python.
        """
        pass


class MetaDT(MetaMLModel):

    def __init__(self, max_depth: int = 3):
        super().__init__(tree.DecisionTreeRegressor, max_depth=max_depth)

    @property
    def name(self) -> str:
        return 'dt'

    def fit(self, data: numpy.ndarray, scores: numpy.ndarray):
        logger.info(f'Fitting meta-model {self.name} on data with shape {data.shape} ...')
        self.model.fit(data, scores)
        return

    def extract_python(self, metric: str, method: str) -> tuple[list[str], str]:
        # noinspection PyProtectedMember
        undefined_feature = tree._tree.TREE_UNDEFINED

        feature_names = [
            Manifold.RATIO_NAMES[i] if i != undefined_feature else "undefined!"
            for i in self.model.tree_.feature
        ]

        # start function definition
        function_name = f'{self.name}_{metric}_{method}'
        code_lines: list[str] = [
            f'def {function_name}(ratios: numpy.ndarray) -> float:',
            f'    {", ".join(Manifold.RATIO_NAMES)} = tuple(ratios)',
        ]

        def extract_lines(node_index: int, indent: str):
            if self.model.tree_.feature[node_index] != undefined_feature:  # internal node

                feature: str = feature_names[node_index]
                threshold: float = self.model.tree_.threshold[node_index]
                left_index: int = self.model.tree_.children_left[node_index]
                right_index: int = self.model.tree_.children_right[node_index]

                # if block
                code_lines.append(f'{indent}if {feature} <= {threshold:.6e}:')
                extract_lines(left_index, indent + '    ')

                # else block
                code_lines.append(f'{indent}else:')
                extract_lines(right_index, indent + '    ')

            else:  # leaf node
                # return
                value: float = self.model.tree_.value[node_index][0][0]
                code_lines.append(f'{indent}return {value:.6e}')

        extract_lines(0, '    ')
        code = '\n'.join(code_lines)

        imports = ['import numpy']

        return imports, code


class MetaLR(MetaMLModel):

    def __init__(self):
        super().__init__(linear_model.LinearRegression)

    @property
    def name(self) -> str:
        return 'lr'

    def fit(self, data: numpy.ndarray, scores: numpy.ndarray):
        logger.info(f'Fitting meta-model {self.name} on data with shape {data.shape} ...')
        self.model.fit(data, scores)
        return

    def extract_python(self, metric: str, method: str) -> tuple[list[str], str]:
        imports = ['import numpy']

        function_name = f'{self.name}_{metric}_{method}'
        coefficients = [f'{float(c):.6e}' for c in self.model.coef_]
        code = '\n'.join([
            f'def {function_name}(ratios: numpy.ndarray) -> float:',
            f'    return float(numpy.dot(numpy.asarray(',
            f'        a=[{", ".join(coefficients)}],',
            f'        dtype=float,',
            f'    ), ratios))'
        ])

        return imports, code
