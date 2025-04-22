import typing as t
from ast import literal_eval
from dataclasses import dataclass

from blendsql.common.constants import DEFAULT_NAN_ANS

# The 'modifier' arg can be either '*' or '+',
#   or any string matching '{\d+}'
ModifierType = t.Union[t.Literal["*", "+"], str, None]


@dataclass
class DataType:
    _name: str
    regex: t.Optional[str]
    modifier: t.Optional[ModifierType]
    _coerce_fn: t.Callable

    @property
    def name(self) -> str:
        if self._name != "list" and self.modifier is not None:
            return f"List[{self._name}]"
        return self._name

    def coerce_fn(self, s: t.Union[str, None]) -> t.Any:
        """Language models output strings.
        We want to coerce their outputs to the DB-friendly type here.
        """
        return self._coerce_fn(s)


def str_to_bool(s: t.Union[str, None]) -> t.Union[bool, str, None]:
    if isinstance(s, str):
        return {
            "t": True,
            "f": False,
            "true": True,
            "false": False,
            "y": True,
            "n": False,
            "yes": True,
            "no": False,
            DEFAULT_NAN_ANS: None,
        }.get(s.lower(), s)
    return s


def str_to_numeric(s: t.Union[str, None]) -> t.Union[float, int, str, None]:
    if not isinstance(s, str):
        return s
    s = s.replace(",", "")
    try:
        casted_s = literal_eval(s)
        assert isinstance(casted_s, (float, int, str))
    except (ValueError, SyntaxError, AssertionError):
        return s
    return casted_s


@dataclass
class DataTypes:
    STR = lambda modifier=None: DataType("str", None, modifier, lambda s: s)
    BOOL = lambda modifier=None: DataType(
        "bool", r"(t|f|true|false|True|False)", modifier, str_to_bool
    )
    INT = lambda modifier=None: DataType("int", r"(\d+)", modifier, str_to_numeric)
    FLOAT = lambda modifier=None: DataType(
        "float", r"(\d+(\.\d+)?)", modifier, str_to_numeric
    )
    ISO_8601_DATE = lambda modifier=None: DataType(
        "date", r"\d{4}-\d{2}-\d{2}", modifier, lambda s: s
    )
    ANY = lambda modifier=None: DataType(
        "Any", None, modifier, lambda s: str_to_numeric(str_to_bool(s))
    )


STR_TO_DATATYPE: t.Dict[str, DataType] = {
    "str": DataTypes.STR(),
    "int": DataTypes.INT(),
    "float": DataTypes.FLOAT(),
    "bool": DataTypes.BOOL(),
    "date": DataTypes.ISO_8601_DATE(),
    "substring": DataTypes.STR(),
    "List[Any]": DataTypes.ANY("*"),
    "List[str]": DataTypes.STR("*"),
    "List[int]": DataTypes.INT("*"),
    "List[float]": DataTypes.FLOAT("*"),
    "List[bool]": DataTypes.BOOL("*"),
    "List[date]": DataTypes.ISO_8601_DATE("*"),
    "List[substring]": DataTypes.STR("*"),
}
