from typing import Literal, Union

from pydantic import BaseModel, Field


class Training(BaseModel):
    """Training feature purpose.

    Specifies a feature used during model training.

    Attributes:
        name (Literal["training"]): Feature purpose discriminator field.
    """

    name: Literal["training"]


class Identifier(BaseModel):
    """Identifier feature purpose.

    Specifies a feature that uniquely identifies samples in the dataset.
    This feature is not used for training or evaluation.
    If an evaluation includes information at the sample level, an identifier feature can be used to
    associate the evaluation results with the original samples.

    Attributes:
        name (Literal["identifier"]): Feature purpose discriminator field.
    """

    name: Literal["identifier"]


class Target(BaseModel):
    """Target feature purpose.

    Specifies a feature that should be predicted from the training features.

    Attributes:
        name (Literal["target"]): Feature purpose discriminator field.
    """

    name: Literal["target"]


class Metadata(BaseModel):
    """Metadata feature purpose.

    Used to provide additional information about each sample.
    This information is not used during training or evaluation.

    Attributes:
        name (Literal["metadata"]): Feature purpose discriminator field.
    """

    name: Literal["metadata"]


class SampleWeight(BaseModel):
    """Sample weight evaluation feature purpose.

    Used during evaluation to weight how much samples should contribute to evaluation metrics.

    Attributes:
        name (Literal["sample_weight"]): Feature purpose discriminator field.
    """

    name: Literal["sample_weight"]


EvaluationFeaturePurpose = Union[SampleWeight]


class Evaluation(BaseModel):
    """Evaluation feature purpose.

    Specifies how samples should be evaluated.

    Attributes:
        name (Literal["evaluation"]): Feature purpose discriminator field.
        type (EvaluationFeaturePurpose): Evaluation feature purpose subtype.
    """

    name: Literal["evaluation"]
    type: EvaluationFeaturePurpose = Field(..., discriminator="name")


class Stratification(BaseModel):
    """Stratification grouping feature purpose.

    Used during training to ensure that samples from different groups
    are evenly distributed between training and evaluation.

    Attributes:
        name (Literal["stratification"]): Feature purpose discriminator field.
    """

    name: Literal["stratification"]


class SubjectWise(BaseModel):
    """Subject-wise grouping feature purpose.

    Used during training to ensure that samples within each group are either
    entirely used for training or evaluation.

    Attributes:
        name (Literal["subject_wise"]): Feature purpose discriminator field.
    """

    name: Literal["subject_wise"]


GroupingFeaturePurpose = Union[Stratification, SubjectWise]


class Grouping(BaseModel):
    """Grouping feature purpose.

    Specifies how groups of samples should be treated during training and evaluation.

    Attributes:
        name (Literal["grouping"]): Feature purpose discriminator field.
        type (GroupingFeaturePurpose): Grouping feature purpose subtype.
    """

    name: Literal["grouping"]
    type: GroupingFeaturePurpose = Field(..., discriminator="name")


class Sensitive(BaseModel):
    """Sensitive feature purpose.

    Specifies that a feature is sensitive, should not be used for training, and may be used during evaluation
    for analysis of fairness or bias.

    Attributes:
        name (Literal["sensitive"]): Feature purpose discriminator field.
    """

    name: Literal["sensitive"]


FeaturePurpose = Union[Training, Identifier, Target, Metadata, Evaluation, Grouping, Sensitive]
