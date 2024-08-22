import logging

import click

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

    print("Surv")

