from enum import Enum, EnumMeta
import json

HF_REPO_ID = "parkervg/blendsql-test-dbs"


class StrInMeta(EnumMeta):
    def __contains__(cls, item):
        return item in cls.__members__.values()


DEFAULT_ANS_SEP = ";"
DEFAULT_NAN_ANS = "-"
DEFAULT_CONTEXT_FORMATTER = lambda df: json.dumps(
    df.to_dict(orient="records"), ensure_ascii=False, indent=4
)
INDENT = lambda n=1: "\t" * n


class IngredientType(str, Enum, metaclass=StrInMeta):
    MAP = "MAP"
    STRING = "STRING"
    QA = "QA"
    JOIN = "JOIN"
    ALIAS = "ALIAS"


class IngredientArgType(str):
    pass


class Subquery(IngredientArgType):
    pass


class ColumnRef(IngredientArgType):
    # '{table}.{column}' syntax
    pass
