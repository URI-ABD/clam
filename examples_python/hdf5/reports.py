import pathlib
import typing

import numpy
import pydantic
from matplotlib import pyplot


class TreeReport(pydantic.BaseModel):
    data_name: str
    metric_name: str
    cardinality: int
    dimensionality: int
    root_name: str
    max_depth: int
    build_time: float


class ClusterReport(pydantic.BaseModel):
    cardinality: int
    depth: int
    name: str
    variant: str
    radius: float
    lfd: float
    indices: typing.Optional[list[int]]
    children: typing.Optional[list[str]]
    ratios: list[float]
    naive_radius: float
    scaled_radius: float


def load_tree(base_path: pathlib.Path) -> list[ClusterReport]:
    return [
        ClusterReport.parse_file(path)
        for path in base_path.iterdir()
        if path.is_file() and 'tree' not in path.name
    ]


# TODO: Update after changes are done in Rust
class RnnReport(pydantic.BaseModel):
    data_name: str
    metric_name: str
    num_queries: int
    num_runs: int
    cardinality: int
    dimensionality: int
    tree_depth: int
    build_time: float
    root_radius: float
    search_radii: numpy.ndarray
    search_times: numpy.ndarray
    outputs: list[list[int]]
    recalls: numpy.ndarray

    # noinspection PyMethodParameters
    @pydantic.validator('search_radii', pre=True)
    def parse_search_radii(search_radii):
        return numpy.array(search_radii, dtype=numpy.float64)

    # noinspection PyMethodParameters
    @pydantic.validator('search_times', pre=True)
    def parse_search_times(search_times):
        return numpy.array(search_times, dtype=numpy.float64)

    # noinspection PyMethodParameters
    @pydantic.validator('recalls', pre=True)
    def parse_recalls(recalls):
        return numpy.array(recalls, dtype=numpy.float64)

    class Config:
        arbitrary_types_allowed = True

    # noinspection DuplicatedCode
    def is_valid(self):
        reasons: list[str] = list()

        if self.num_queries != len(self.search_radii):
            reasons.append(f'self.num_queries != len(self.search_radii): {self.num_queries} != {len(self.search_radii)}')

        if self.num_queries != len(self.search_times):
            reasons.append(f'self.num_queries != len(self.search_times): {self.num_queries} != {len(self.search_times)}')

        if self.num_queries != len(self.outputs):
            reasons.append(f'self.num_queries != len(self.outputs): {self.num_queries} != {len(self.outputs)}')

        if self.num_queries != len(self.recalls):
            reasons.append(f'self.num_queries != len(self.recalls): {self.num_queries} != {len(self.recalls)}')

        for i, samples in enumerate(self.search_times, start=1):
            if self.num_runs != len(samples):
                reasons.append(f'{i}/{self.num_queries}: self.num_runs != samples.len(): {self.num_runs} != {len(samples)}')

        return reasons

    @property
    def output_sizes(self) -> numpy.ndarray:
        return numpy.asarray(list(map(len, self.outputs)))

    # noinspection PyMethodMayBeStatic
    def _plot_xy(self, ax: pyplot.Axes, x: numpy.ndarray, y: numpy.ndarray):
        ax.scatter(x, y, s=0.2)
        return

    def _plot_search_radii_vs_search_times(self, ax: pyplot.Axes) -> pyplot.Axes:
        self._plot_xy(ax, self.search_radii, numpy.mean(self.search_times, axis=1))
        ax.set_yscale('log')
        ax.set_xlabel('Search Radius')
        ax.set_ylabel('Search Time (sec)')
        return ax

    def _plot_search_radii_vs_output_sizes(self, ax: pyplot.Axes) -> pyplot.Axes:
        self._plot_xy(ax, self.search_radii, self.output_sizes)
        ax.set_xlabel('Search Radius')
        ax.set_ylabel('Output Size')
        return ax

    def _plot_search_radii_vs_recalls(self, ax: pyplot.Axes) -> pyplot.Axes:
        self._plot_xy(ax, self.search_radii, self.recalls)
        ax.set_xlabel('Search Radius')
        ax.set_ylabel('Recall')
        ax.set_ylim(0, 1.1)
        return ax

    def _plot_output_sizes_vs_search_times(self, ax: pyplot.Axes) -> pyplot.Axes:
        self._plot_xy(ax, self.output_sizes, numpy.mean(self.search_times, axis=1))
        ax.set_xlabel('Output Size')
        ax.set_ylabel('Search Time (sec)')
        return ax

    def _plot_output_sizes_vs_recalls(self, ax: pyplot.Axes) -> pyplot.Axes:
        self._plot_xy(ax, self.output_sizes, self.recalls)
        ax.set_xlabel('Output Size')
        ax.set_ylabel('Recall')
        return ax

    # noinspection DuplicatedCode
    def plot(
            self,
            show: bool,
            output_dir: pathlib.Path,
    ):

        figure, ((ax00, ax01), (ax10, ax11)) = pyplot.subplots(2, 2, figsize=(8.5, 11))
        # figure.suptitle(f'Ranged Nearest Neighbors Search')
        figure.suptitle(f'{self.data_name}, {self.metric_name}, ({self.cardinality}, {self.dimensionality})')

        self._plot_search_radii_vs_search_times(ax00)
        self._plot_output_sizes_vs_recalls(ax01)
        self._plot_search_radii_vs_recalls(ax10)
        self._plot_output_sizes_vs_search_times(ax11)

        if show:
            pyplot.show()
        else:
            figure.savefig(output_dir.joinpath(f'rnn-search-{self.data_name}-{self.metric_name}.png'), dpi=300)

        pyplot.close(figure)

        return
