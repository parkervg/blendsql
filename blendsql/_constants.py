from enum import Enum, EnumMeta, auto
from dataclasses import dataclass

HF_REPO_ID = "parkervg/blendsql-test-dbs"


class StrInMeta(EnumMeta):
    def __contains__(cls, item):
        return item in cls.__members__.values()


DEFAULT_ANS_SEP = ";"
DEFAULT_NAN_ANS = "-"
VALUE_BATCH_SIZE = 5


class IngredientType(str, Enum, metaclass=StrInMeta):
    MAP = auto()
    STRING = auto()
    QA = auto()
    JOIN = auto()


@dataclass
class IngredientKwarg:
    QUESTION = "question"
    CONTEXT = "context"
    VALUES = "values"
    OPTIONS = "options"
    MODEL = "model"
