from typing import NewType, Literal, Union
from enum import Enum, EnumMeta
from dataclasses import dataclass

HF_REPO_ID = "parkervg/blendsql-test-dbs"


class StrInMeta(EnumMeta):
    def __contains__(cls, item):
        return item in cls.__members__.values()


DEFAULT_ANS_SEP = ";"
DEFAULT_NAN_ANS = "-"

# The 'modifier' arg can be either '*' or '+',
#   or any string matching '{\d+}'
ModifierType = NewType("modifier", Union[Literal["*", "+"], str, None])


class IngredientType(str, Enum, metaclass=StrInMeta):
    MAP = "MAP"
    STRING = "STRING"
    QA = "QA"
    JOIN = "JOIN"
    ALIAS = "ALIAS"


@dataclass
class IngredientKwarg:
    QUESTION: str = "question"
    CONTEXT: str = "context"
    VALUES: str = "values"
    OPTIONS: str = "options"
    REGEX: str = "regex"
    MODEL: str = "model"
