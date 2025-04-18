import typing as t
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
ModifierType = t.Union[t.Literal["*", "+"], str, None]


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
    OUTPUT_TYPE: str = "output_type"
    EXAMPLE_OUTPUTS: str = "example_outputs"


@dataclass
class DataType:
    _name: str
    regex: t.Optional[str]
    modifier: t.Optional[ModifierType]

    @property
    def name(self) -> str:
        if self._name != "list" and self.modifier is not None:
            return f"List[{self._name}]"
        return self._name


@dataclass
class DataTypes:
    BOOL = lambda modifier=None: DataType(
        "bool", "(t|f|true|false|True|False)", modifier
    )
    INT = lambda modifier=None: DataType("int", "(\d+)", modifier)
    FLOAT = lambda modifier=None: DataType("float", "(\d+(\.\d+)?)", modifier)
    ISO_8601_DATE = lambda modifier=None: DataType(
        "date", "\d{4}-\d{2}-\d{2}", modifier
    )
    STR = lambda modifier=None: DataType("str", None, modifier)
    LIST = lambda modifier="*": DataType("list", None, modifier)
    ANY = lambda modifier=None: DataType("Any", None, modifier)


STR_TO_DATATYPE: t.Dict[str, DataType] = {
    "str": DataTypes.STR(),
    "int": DataTypes.INT(),
    "float": DataTypes.FLOAT(),
    "bool": DataTypes.BOOL(),
    "date": DataTypes.ISO_8601_DATE(),
    "substring": DataTypes.STR(),
    "List[str]": DataTypes.STR("*"),
    "List[int]": DataTypes.INT("*"),
    "List[float]": DataTypes.FLOAT("*"),
    "List[bool]": DataTypes.BOOL("*"),
    "List[date]": DataTypes.ISO_8601_DATE("*"),
    "List[substring]": DataTypes.STR("*"),
}
