import json
import logging

import click
from pydantic import TypeAdapter
from rich.pretty import pprint

from surv.models.question import Question
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
    questions_filepath = settings.data_dirpath / "questions.json"
    with questions_filepath.open("r") as file:
        questions_json = json.load(file)
        type_adapter = TypeAdapter(list[Question])
        questions = type_adapter.validate_python(questions_json)
        pprint(questions)
