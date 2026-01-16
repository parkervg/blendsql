from ast import literal_eval
from dataclasses import dataclass

from blendsql.common.constants import DEFAULT_NAN_ANS
from blendsql.common.typing import DataType
from blendsql.db import Database


def unquote(s):
    for quote in ['"', "'"]:
        s = s.removeprefix(quote).removesuffix(quote)
    return s


def str_to_bool(s: str | None, _: Database | None) -> bool | str | None:
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


def str_to_numeric(s: str | None, _: Database | None) -> float | int | None:
    if not isinstance(s, str):
        return s
    s = s.replace(",", "")
    try:
        casted_s = literal_eval(s)
        assert isinstance(casted_s, (float, int))
    except (ValueError, SyntaxError, AssertionError):
        return None
    return casted_s


def str_to_date(s: str | None, db: Database | None) -> str | None:
    if db.__class__.__name__ == "DuckDB":
        return f"DATE '{s}'"
    return s


def str_to_str(s: str | None, _: Database | None) -> str | None:
    if not isinstance(s, str):
        return s
    return unquote(s)


@dataclass
class DataTypes:
    STR = lambda quantifier=None: DataType(
        atomic_type="str",
        regex=None,
        quantifier=quantifier,
        _coerce_fn=str_to_str,
        requires_quotes=True,
    )
    SUBSTRING = lambda quantifier=None: DataType(
        atomic_type="substring",
        regex=None,
        quantifier=quantifier,
        _coerce_fn=str_to_str,
        requires_quotes=True,
    )
    BOOL = lambda quantifier=None: DataType(
        atomic_type="bool",
        regex=r"(t|f|true|false|True|False)",
        quantifier=quantifier,
        _coerce_fn=str_to_bool,
    )
    INT = lambda quantifier=None: DataType(
        atomic_type="int",
        regex=r"-?(\d+)",
        quantifier=quantifier,
        _coerce_fn=str_to_numeric,
    )
    FLOAT = lambda quantifier=None: DataType(
        atomic_type="float",
        regex="-?(\d+(\.\d+)?)",
        quantifier=quantifier,
        _coerce_fn=str_to_numeric,
    )
    NUMERIC = lambda quantifier=None: DataType(
        atomic_type="Union[int, float]",
        regex=r"(\d+(\.\d+)?)",
        quantifier=quantifier,
        _coerce_fn=str_to_numeric,
    )
    ISO_8601_DATE = lambda quantifier=None: DataType(
        atomic_type="NewType(DateString_YYYY_MM_DD, str)",
        regex=r"\d{4}-\d{2}-\d{2}",
        quantifier=quantifier,
        _coerce_fn=str_to_date,
        requires_quotes=True,
    )
    ANY = lambda quantifier=None: DataType(
        atomic_type="Any",
        regex=None,
        quantifier=quantifier,
        _coerce_fn=lambda s, _: s,  # Let the DBMS transform, if it allows
    )


STR_TO_DATATYPE: dict[str, DataType] = {
    "str": DataTypes.STR(),
    "int": DataTypes.INT(),
    "float": DataTypes.FLOAT(),
    "bool": DataTypes.BOOL(),
    "date": DataTypes.ISO_8601_DATE(),
    "substring": DataTypes.SUBSTRING(),
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
