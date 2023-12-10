"""Parser for the scaling results of the Cakes search."""

import json
import pathlib
import typing

import pandas
import pydantic


class Report(pydantic.BaseModel):
    """Report of the scaling results of the Cakes search."""

    dataset: str
    metric: str
    base_cardinality: int
    dimensionality: int
    num_queries: int
    error_rate: float
    ks: list[int]
    csv_path: pathlib.Path = pathlib.Path(".").resolve()

    @staticmethod
    def from_json(json_path: pathlib.Path) -> "Report":
        """Load the report from a JSON file."""
        with json_path.open("r") as json_file:
            contents: dict[str, typing.Any] = json.load(json_file)
            contents["csv_path"] = json_path.parent.joinpath(contents.pop("csv_name"))
            return Report(**contents)

    def to_pandas(self) -> pandas.DataFrame:
        """Read the CSV file into a pandas DataFrame."""
        return pandas.read_csv(self.csv_path)
