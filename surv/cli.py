import logging

import click
from rich.pretty import pprint

from surv.algo.constraints import Constraint
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
    feature = evaluator.evaluate(dataset, constraints)
    pprint(feature)


if __name__ == "__main__":
    main()
