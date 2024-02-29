from enum import Enum, EnumMeta, auto
from dataclasses import dataclass

HF_REPO_ID = "parkervg/blendsql-test-dbs"


class StrInMeta(EnumMeta):
    def __contains__(cls, item):
        return item in cls.__members__.values()


OPENAI_COMPLETE_LLM = ["text-davinci-003"]
OPENAI_CHAT_LLM = [
    "gpt-4",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-35-turbo",
    "gpt-35-turbo-0613",
    "gpt-35-turbo-16k-0613",
    "gpt-35-turbo-instruct-0914",
]

DEFAULT_ANS_SEP = ";"
DEFAULT_NAN_ANS = "-"
VALUE_BATCH_SIZE = 10


class IngredientType(str, Enum, metaclass=StrInMeta):
    MAP = auto()
    STRING = auto()
    QA = auto()
    JOIN = auto()


@dataclass
class IngredientKwarg:
    QUESTION = "question"
    CONTEXT = "context"
    OPTIONS = "options"
    MODEL = "model"
