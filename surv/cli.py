import logging

import click
import numpy as np
from rich.logging import RichHandler

from surv.algo.constraints import Constraint, EqConstraint
from surv.algo.evaluator import Evaluator
from surv.models.dataset import Dataset
from surv.settings import Settings

logger = logging.getLogger(__name__)


@click.group()
def main() -> None:
    """Run main CLI entrypoint."""


def set_logger_config(info: bool, debug: bool) -> None:
    """Set the logger configuration."""
    handlers = [RichHandler(rich_tracebacks=True)]
    log_format = "%(message)s"

    if info:
        logging.basicConfig(level=logging.INFO, handlers=handlers, format=log_format)
    if debug:
        logging.basicConfig(level=logging.DEBUG, handlers=handlers, format=log_format)


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
    feature_info_filepath = settings.data_dirpath / "features.json"
    dataset = Dataset.from_files(tabular_filepath, feature_info_filepath)
    logger.debug("Dataset: %s", dataset)
    logger.debug("Number of samples: %s", dataset.n_samples)
    logger.debug("Number of features: %s", dataset.n_features)
    rng = np.random.default_rng()

    evaluator = Evaluator()
    constraints: list[Constraint] = []
    while len(constraints) < dataset.n_training_features:
        result = evaluator.evaluate(dataset, constraints)
        feature = result.feature
        information_gain = result.information_gain

        logger.info("Found feature with highest information gain (%f): %s", information_gain, feature)

        if information_gain == 0.0:
            logger.info("No information gain.")
            break

        if information_gain == settings.information_gain_threshold:
            logger.info("Information gain threshold reached.")
            break

        # Randomly choose a value from the same distribution as the existing dataset.
        column = dataset.get_column(feature.name)
        rand_int = rng.integers(0, column.shape[0])
        value = column[rand_int]
        constraint = EqConstraint(feature=feature, value=value)
        logger.info("Chose constraint: %s", constraint)
        constraints.append(constraint)

    logger.info("Final constraints: %s", constraints)


if __name__ == "__main__":
    main()
