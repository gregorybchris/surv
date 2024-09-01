# Surv

Surv is a dynamic survey program. Based on your answers to the questions, Surv will select the next question that maximizes expected information gain. This means that in most cases you don't need to fill out every question in the survey.

## Installation

[Poetry](https://python-poetry.org/) is a requirement

```bash
poetry install
```

## CLI

```bash
export SURV_DATA_DIRPATH="<path-to-data>"
surv run <dataset-name>
```

## Run tests

```bash
pytest tests
```
