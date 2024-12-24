<div align="center">
  <h1>Surv</h1>

  <p>
    <strong>Dynamic survey generator</strong>
  </p>

  <hr />
</div>

## About

Based on your answers to survey questions, Surv will select the next question that maximizes expected information gain. This means that in most cases you don't need to fill out every question in the survey before your result is known to a high degree of confidence.

You must have a feature you are trying to predict to use Surv. You need something information gain can be relative to. You can use Surv to collect a dataset, but if you don't have a seed dataset already collected, then each participant will need to take the survey in full.

## Installation

[Poetry](https://python-poetry.org/) is a requirement

```bash
poetry install
```

## CLI usage

This command will start up an interactive session where you can fill out a survey and see how the entropy decreases as Surv becomes more certain about your most likely classification.

```bash
export SURV_DATA_DIRPATH="<data-dirpath>"
poetry run surv run <dataset-name>
```

### Example

The housing market example is loosely based on the [Boston housing dataset](https://scikit-learn.org/0.16/modules/generated/sklearn.datasets.load_boston.html). By filling out information about a hypothetical house (like the yard size, garage status, and presence of mold) Surv will decrease its uncertainty about the value of the house.

```bash
export SURV_DATA_DIRPATH="tests/algo/data"
poetry run surv run house
```

## Dataset representation

Surv also comes with a dataset feature metadata system. Feature types and feature purposes help you to avoid common bugs when processing structured data with heterogeneous features.

### Feature types

Track metadata like the cardinality of categorical features, whether features are continuous or discrete, and whether values should be treated like integers or floating point numbers.

| Feature Type | Feature Subtype |
| ------------ | --------------- |
| Categorical  | Binary          |
| Categorical  | Multiclass      |
| Numeric      | Ordinal         |
| Numeric      | Interval        |
| Numeric      | Ratio           |
| Datetime     | -               |
| Text         | -               |

### Feature purposes

Tagging the purpose of features can help you filter down features for certain use cases like training or identifying a unique sample across multiple datasets.

| Feature Purpose |
| --------------- |
| Training        |
| Identifier      |
| Target          |
| Metadata        |
| SampleWeight    |
| Evaluation      |
| Stratification  |
| SubjectWise     |
| Grouping        |
| Sensitive       |

## Running tests

```bash
pytest tests
```
