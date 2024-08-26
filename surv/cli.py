import logging

import click
from rich.pretty import pprint

from surv.algo.constraints import Constraint, EqConstraint
from surv.algo.evaluator import Evaluator
from surv.models.dataset import Dataset
from surv.settings import Settings


@click.group()
def main() -> None:
    """Run main CLI entrypoint."""


def set_logger_config(info: bool, debug: bool) -> None:
    """Set the logger configuration."""
    if info:
        logging.basicConfig(level=logging.INFO)
    if debug:
        logging.basicConfig(level=logging.DEBUG)


@main.command(name="run")
@click.option("--info", is_flag=True)
@click.option("--debug", is_flag=True)
def run_command(
    info: bool = False,
    debug: bool = False,
) -> None:
    """Run the CLI."""
    set_logger_config(info, debug)

    settings = Settings()

    tabular_filepath = settings.data_dirpath / "tabular.csv"
    metadata_filepath = settings.data_dirpath / "metadata.json"
    dataset = Dataset.from_files(tabular_filepath, metadata_filepath)
    pprint(dataset)

    evaluator = Evaluator()
    constraints: list[Constraint] = []
    while len(constraints) < dataset.n_features:
        result = evaluator.evaluate(dataset, constraints)
        feature = result.feature
        information_gain = result.information_gain
        if information_gain == settings.information_gain_threshold:
            print("Information gain threshold reached.")
            break
        constraint = EqConstraint(feature=feature, value="0")
        print("Feature with highest information gain: ")
        pprint(feature)
        constraints.append(constraint)


if __name__ == "__main__":
    main()
