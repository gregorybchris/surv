<div align="center">
  <h1>Surv</h1>

  <p>
    <strong>Adaptive survey engine</strong>
  </p>

  <hr />
</div>

## About

Nobody likes to fill out long surveys, especially ones with obviously redundant questions. Surv is designed to keep participants engaged by minimizing the number of questions while maximizing the information gained from each question.

Survey length is especially important when the participant has a choice in whether to complete the survey. When filling out a medical intake form you may have no choice but to answer 100 questions, but for use cases like market research, product usability studies, job satisfaction surveys, or political polls, the length and precision of the survey can significantly affect the probability of survey completion. And low completion rate can lead to sample bias and decreased survey validity.

As you answer questions on a Surv survey, the expected information gain is recalculated, determining which question to present next. In most cases you don't need to fill out every question in the survey before results are known to a high degree of confidence.

The paradox of the adaptive survey is that while each participant spends less time taking the survey, you are able to include more total questions in the survey, including questions that do not apply to a large portion of the population, but are highly informative for some individuals.

## Installation

Install using [uv](https://docs.astral.sh/uv)

```bash
uv sync
```

## CLI usage

This command will start up an interactive session where you can fill out a survey and see how the entropy decreases as Surv becomes more certain about your most likely classification.

```bash
export SURV_DATA_DIRPATH="<data-dirpath>"
uv run surv run <dataset-name> --info
```

### Example

The housing market example is loosely based on the [Boston housing dataset](https://scikit-learn.org/0.16/modules/generated/sklearn.datasets.load_boston.html). By filling out information about a hypothetical house (like the yard size, garage status, and presence of mold) Surv will decrease its uncertainty about the value of the house.

```bash
export SURV_DATA_DIRPATH="<path-to>/surv/tests/algo/data"
uv run surv run house --info
```

## Dataset representation

Surv also comes with a dataset feature metadata system. Feature types and feature purposes help you to avoid common bugs when processing structured data with heterogeneous features.

### Feature types

Track metadata like the cardinality of categorical features, whether features are continuous or discrete, and whether values should be treated like integers or floating point numbers.

| Feature Type | Feature Subtype | Description                                                                                      |
| ------------ | --------------- | ------------------------------------------------------------------------------------------------ |
| Categorical  | Binary          | Two possible values, e.g. yes/no, true/false                                                     |
| Categorical  | Multiclass      | More than two possible values, e.g. red/green/blue                                               |
| Numeric      | Ordinal         | Values have a meaningful order, e.g. low/medium/high                                             |
| Numeric      | Interval        | Values have a meaningful order and equal intervals, e.g. temperature in Celsius                  |
| Numeric      | Ratio           | Values have a meaningful order, equal intervals, and a true zero point, e.g. weight in kilograms |
| Datetime     | -               | Date and time values                                                                             |
| Text         | -               | Free-form text, e.g. comments or descriptions                                                    |

### Feature purposes

Tagging the purpose of features can help you filter down features for certain use cases like training or identifying a unique sample across multiple datasets.

| Feature Purpose | Description                                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------ |
| Training        | Features used as inputs during training                                                                            |
| Target          | Features that a model is trained to predict, e.g. house price or customer satisfaction                             |
| Identifier      | Features that uniquely identify a sample, e.g. user ID or survey response ID                                       |
| Metadata        | Arbitrary metadata features                                                                                        |
| SampleWeight    | Features that weight samples by importance for either training or evaluation                                       |
| Stratification  | Features used to ensure even splits of data across different groups                                                |
| SubjectWise     | Features that group data by subject to ensure within subject samples are not used for both training and validation |
| Sensitive       | Features that should not be used for training, but may be used for evaluation to ensure fairness or evaluate bias  |

## Running tests

```bash
pytest tests
```
