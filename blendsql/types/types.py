from ast import literal_eval
from dataclasses import dataclass
import re

from blendsql.common.constants import DEFAULT_NAN_ANS
from blendsql.common.typing import DataType, QuantifierType


def str_to_bool(s: str | None) -> bool | str | None:
    return {
        "t": True,
        "f": False,
        "true": True,
        "false": False,
        "y": True,
        "n": False,
        "yes": True,
        "no": False,
        "1": True,
        "0": False,
        DEFAULT_NAN_ANS: None,
    }.get(s.lower(), None)


def str_to_numeric(s: str | None) -> float | int | None:
    if not isinstance(s, str):
        return s
    s = s.replace(",", "")
    try:
        casted_s = literal_eval(s)
        assert isinstance(casted_s, (float, int))
    except (ValueError, SyntaxError, AssertionError):
        return None
    return casted_s


def maybe_str_to_str_list(
    s: str | None, quantifier: QuantifierType
) -> list[str | None]:
    if quantifier is None:
        return s
    try:
        return literal_eval(s)
    except:
        # Sometimes we need to first escape single quotes
        # E.g. in ['Something's wrong here']
        s_fixed = re.sub(r"(\w)'(\w)", r"\1\\'\2", s)
        return literal_eval(s_fixed)


@dataclass
class DataTypes:
    STR = lambda quantifier=None: DataType(
        "str", None, quantifier, lambda s: maybe_str_to_str_list(s, quantifier)
    )
    BOOL = lambda quantifier=None: DataType(
        "bool", r"(t|f|true|false|True|False)", quantifier, str_to_bool
    )
    INT = lambda quantifier=None: DataType(
        "int", r"-?(\d+)", quantifier, str_to_numeric
    )
    FLOAT = lambda quantifier=None: DataType(
        "float", r"-?(\d+(\.\d+)?)", quantifier, str_to_numeric
    )
    NUMERIC = lambda quantifier=None: DataType(
        "Union[int, float]", r"(\d+(\.\d+)?)", quantifier, str_to_numeric
    )
    ISO_8601_DATE = lambda quantifier=None: DataType(
        "date", r"\d{4}-\d{2}-\d{2}", quantifier, lambda s: s
    )
    ANY = lambda quantifier=None: DataType(
        "Any", None, quantifier, lambda s: s  # Let the DBMS transform, if it allows
    )


STR_TO_DATATYPE: dict[str, DataType] = {
    "str": DataTypes.STR(),
    "int": DataTypes.INT(),
    "float": DataTypes.FLOAT(),
    "bool": DataTypes.BOOL(),
    "date": DataTypes.ISO_8601_DATE(),
    "substring": DataTypes.STR(),
    "any": DataTypes.ANY(),
    "list[Any]": DataTypes.ANY("*"),
    "list[str]": DataTypes.STR("*"),
    "list[int]": DataTypes.INT("*"),
    "list[float]": DataTypes.FLOAT("*"),
    "list[bool]": DataTypes.BOOL("*"),
    "list[date]": DataTypes.ISO_8601_DATE("*"),
    "list[substring]": DataTypes.STR("*"),
}

# Align different database types to our type system
DB_TYPE_TO_STR = {
    "BIGINT": "int",
    "VARCHAR": "str",
    "BOOLEAN": "bool",
    "FLOAT": "float",
    "HUGEINT": "int",
    "INTEGER": "int",
    "SMALLINT": "int",
    "TEXT": "str",
    "REAL": "float",
    "DATE": "date",
}
