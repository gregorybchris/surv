from typing import Literal, Union

from pydantic import BaseModel, Field


class Binary(BaseModel):
    """Binary categorical variable.

    Attributes:
        name (Literal["binary"]): Feature type discriminator field.
        positive_class (str): Positive class name.
    """

    name: Literal["binary"]
    positive_class: str


class Multiclass(BaseModel):
    """Multiclass categorical variable.

    Attributes:
        name (Literal["multiclass"]): Feature type discriminator field.
    """

    name: Literal["multiclass"]


CategoricalFeatureType = Union[Binary, Multiclass]


class Categorical(BaseModel):
    """Categorical variable.

    Attributes:
        name (Literal["categorical"]): Feature type discriminator field.
        type (CategoricalFeatureType): Categorical variable subtype.
        categories (Optional[str]): List of categories.
    """

    name: Literal["categorical"]
    type: CategoricalFeatureType = Field(..., discriminator="name")
    categories: list[str]


class Ordinal(BaseModel):
    """Ordinal numeric variable.

    Attributes:
        name (Literal["ordinal"]): Feature type discriminator field.
    """

    name: Literal["ordinal"]


class Interval(BaseModel):
    """Interval numeric variable.

    Attributes:
        name (Literal["interval"]): Feature type discriminator field.
    """

    name: Literal["interval"]


class Ratio(BaseModel):
    """Ratio numeric variable.

    Attributes:
        name (Literal["ratio"]): Feature type discriminator field.
    """

    name: Literal["ratio"]


NumericFeatureType = Union[Ordinal, Interval, Ratio]


class Numeric(BaseModel):
    """Numeric variable.

    Attributes:
        name (Literal["numeric"]): Feature type discriminator field.
        type (NumericFeatureType): Numeric variable subtype.
    """

    name: Literal["numeric"]
    type: NumericFeatureType = Field(..., discriminator="name")


class Datetime(BaseModel):
    """Datetime variable.

    Attributes:
        name (Literal["datetime"]): Feature type discriminator field.
    """

    name: Literal["datetime"]


class Text(BaseModel):
    """Text variable.

    Attributes:
        name (Literal["text"]): Feature type discriminator field.
    """

    name: Literal["text"]


FeatureType = Union[Categorical, Numeric, Datetime, Text]
