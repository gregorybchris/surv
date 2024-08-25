from typing import Literal, Union

from pydantic import BaseModel, Field


class Binary(BaseModel):
    """Binary categorical variable.

    Attributes:
        type (Literal["binary"]): Discriminator field.
    """

    type: Literal["binary"]
    positive_class: str


class Multiclass(BaseModel):
    """Multiclass categorical variable.

    Attributes:
        type (Literal["multiclass"]): Discriminator field.
    """

    type: Literal["multiclass"]


class Categorical(BaseModel):
    """Categorical variable.

    Attributes:
        type (Literal["categorical"]): Discriminator field.
        metadata (Union[Binary, Multiclass]): Categorical variable metadata.
        positive_class (Optional[str]): Positive class.
    """

    type: Literal["categorical"]
    metadata: Union[Binary, Multiclass] = Field(..., discriminator="type")
    classes: list[str]


class Ordinal(BaseModel):
    """Ordinal numeric variable.

    Attributes:
        type (Literal["ordinal"]): Discriminator field.
    """

    type: Literal["ordinal"]


class Interval(BaseModel):
    """Interval numeric variable.

    Attributes:
        type (Literal["interval"]): Discriminator field.
    """

    type: Literal["interval"]


class Ratio(BaseModel):
    """Ratio numeric variable.

    Attributes:
        type (Literal["ratio"]): Discriminator field.
    """

    type: Literal["ratio"]


class Numeric(BaseModel):
    """Numeric variable.

    Attributes:
        type (Literal["numeric"]): Discriminator field.
        metadata (Union[Ordinal, Interval, Ratio]): Numeric variable metadata.
    """

    metadata: Union[Ordinal, Interval, Ratio] = Field(..., discriminator="type")
    type: Literal["numeric"]


class Datetime(BaseModel):
    """Datetime variable.

    Attributes:
        type (Literal["datetime"]): Discriminator field.
    """

    type: Literal["datetime"]


class Text(BaseModel):
    """Text variable.

    Attributes:
        type (Literal["text"]): Discriminator field.
    """

    type: Literal["text"]


class FeatureType(BaseModel):
    """Feature type model.

    Attributes:
        data (Union[Categorical, Numeric, Datetime, Text]): Feature type metadata.
    """

    metadata: Union[Categorical, Numeric, Datetime, Text] = Field(..., discriminator="type")
