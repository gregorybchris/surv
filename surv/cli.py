import logging
from typing import Optional

import click
from rich.logging import RichHandler

from surv.algo.constraints import Constraint, EqConstraint
from surv.algo.evaluator import Evaluator
from surv.models.dataset import Dataset
from surv.models.feature import Feature
from surv.models.feature_types import Categorical, Datetime, Numeric, Text
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
@click.argument("dataset-name")
@click.option("--info", is_flag=True)
@click.option("--debug", is_flag=True)
def run_command(
    dataset_name: str,
    info: bool = False,
    debug: bool = False,
) -> None:
    """Run the CLI."""
    set_logger_config(info, debug)

    settings = Settings()

    dataset_dirpath = settings.data_dirpath / dataset_name
    tabular_filepath = dataset_dirpath / "tabular.csv"
    feature_info_filepath = dataset_dirpath / "features.json"
    dataset = Dataset.from_files(tabular_filepath, feature_info_filepath)
    logger.debug("Dataset: %s", dataset)
    logger.debug("Number of samples: %s", dataset.n_samples)
    logger.debug("Number of features: %s", dataset.n_features)

    evaluator = Evaluator()
    constraints: list[Constraint] = []
    while len(constraints) < dataset.n_training_features:
        result = evaluator.evaluate(dataset, constraints)
        feature = result.feature
        information_gain = result.information_gain

        logger.info("Found feature with highest information gain (%f): %s", information_gain, feature.name)

        if information_gain == 0.0:
            logger.info("No information gain.")
            break

        if information_gain == settings.information_gain_threshold:
            logger.info("Information gain threshold reached.")
            break

        constraint = accept_input(feature)
        constraints.append(constraint)

    logger.info("Final constraints: %s", constraints)

    target_feature = dataset.get_feature(dataset.feature_info.target_feature_name)
    target_constraint = accept_input(target_feature)

    logger.info("Target constraint: %s", target_constraint)

    print_survey_summary(constraints, target_constraint)


def print_survey_summary(constraints: list[Constraint], target_constraint: Constraint) -> None:
    """Print the survey summary."""
    print("-----")
    print("Survey summary:")
    for constraint in constraints:
        match constraint:
            case EqConstraint(feature=feature, value=value):
                print(f"Question: {feature.name} -> {value}")
            case _:
                raise NotImplementedError

    match target_constraint:
        case EqConstraint(feature=feature, value=value):
            print(f"Question: {feature.name} -> {value}")
        case _:
            raise NotImplementedError


def get_synonym_input(input_str: str, categories: list[str]) -> Optional[str]:
    """Get the synonym input."""
    synonyms = {
        "y": "yes",
        "n": "no",
    }
    if input_str in synonyms and synonyms[input_str] in categories:
        return synonyms[input_str]
    return None


def accept_input(feature: Feature) -> Constraint:
    """Accept input from the user."""
    if feature.metadata is None:
        msg = "Feature metadata is required to run survey."
        raise ValueError(msg)

    match feature.type:
        case Categorical():
            categories = feature.type.classes
            print(f"Question: {feature.metadata.question}")
            num_map = {str(i + 1): category for i, category in enumerate(categories)}
            for n, category in num_map.items():
                print(f"{n}: {category}")

            while True:
                print("> ", end="")
                input_str = input()
                if input_str in num_map:
                    value = num_map[input_str]
                    break
                if input_str in categories:
                    value = input_str
                    break
                synonym = get_synonym_input(input_str, categories)
                if synonym is not None:
                    value = synonym
                    break
                print(f"Invalid input. Please select a number between 1 and {len(num_map)}.")

            assert value in categories
            return EqConstraint(feature=feature, value=value)
        case Numeric():
            raise NotImplementedError
        case Datetime():
            raise NotImplementedError
        case Text():
            raise NotImplementedError


if __name__ == "__main__":
    main()
