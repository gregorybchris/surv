import logging

import click
from rich.pretty import pprint

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

    responses_filepath = settings.data_dirpath / "responses.csv"
    questions_filepath = settings.data_dirpath / "questions.json"
    dataset = Dataset.from_files(questions_filepath, responses_filepath)
    pprint(dataset)


if __name__ == "__main__":
    main()
