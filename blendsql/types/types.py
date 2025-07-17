import typing as t
from ast import literal_eval
from dataclasses import dataclass

from blendsql.common.constants import DEFAULT_NAN_ANS

# The 'quantifier' arg can be either '*' or '+',
#   or any string matching '{\d(,\d)?}'
QuantifierType = t.Union[t.Literal["*", "+"], str, None]


@dataclass
class DataType:
    _name: str
    regex: t.Optional[str]
    quantifier: t.Optional[QuantifierType]
    _coerce_fn: t.Callable

    @property
    def name(self) -> str:
        if self._name != "list" and self.quantifier is not None:
            return f"List[{self._name}]"
        return self._name

    def coerce_fn(self, s: t.Union[str, None]) -> t.Any:
        """Language models output strings.
        We want to coerce their outputs to the DB-friendly type here.
        """
        return self._coerce_fn(s)


def str_to_bool(s: t.Union[str, None]) -> t.Union[bool, str, None]:
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


def str_to_numeric(s: t.Union[str, None]) -> t.Union[float, int, None]:
    if not isinstance(s, str):
        return s
    s = s.replace(",", "")
    try:
        casted_s = literal_eval(s)
        assert isinstance(casted_s, (float, int))
    except (ValueError, SyntaxError, AssertionError):
        return None
    return casted_s


@dataclass
class DataTypes:
    STR = lambda quantifier=None: DataType("str", None, quantifier, lambda s: s)
    BOOL = lambda quantifier=None: DataType(
        "bool", r"(t|f|true|false|True|False)", quantifier, str_to_bool
    )
    INT = lambda quantifier=None: DataType("int", r"(\d+)", quantifier, str_to_numeric)
    FLOAT = lambda quantifier=None: DataType(
        "float", r"(\d+(\.\d+)?)", quantifier, str_to_numeric
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


STR_TO_DATATYPE: t.Dict[str, DataType] = {
    "str": DataTypes.STR(),
    "int": DataTypes.INT(),
    "float": DataTypes.FLOAT(),
    "bool": DataTypes.BOOL(),
    "date": DataTypes.ISO_8601_DATE(),
    "substring": DataTypes.STR(),
    "Any": DataTypes.ANY(),
    "List[Any]": DataTypes.ANY("*"),
    "List[str]": DataTypes.STR("*"),
    "List[int]": DataTypes.INT("*"),
    "List[float]": DataTypes.FLOAT("*"),
    "List[bool]": DataTypes.BOOL("*"),
    "List[date]": DataTypes.ISO_8601_DATE("*"),
    "List[substring]": DataTypes.STR("*"),
}
