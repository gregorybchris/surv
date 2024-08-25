from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings.

    Attributes:
        data_dirpath (Path): Path to directory containing data files.
    """

    data_dirpath: Path = Field(alias="surv_data_dirpath")
