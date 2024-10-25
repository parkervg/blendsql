from typing import NewType, Literal, Union, Optional, Dict
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
    OUTPUT_TYPE: str = "output_type"
    EXAMPLE_OUTPUTS: str = "example_outputs"


@dataclass
class DataType:
    _name: str
    regex: Optional[str]
    modifier: Optional[ModifierType]

    @property
    def name(self) -> str:
        if self._name != "list" and self.modifier is not None:
            return f"List[{self._name}]"
        return self._name


@dataclass
class DataTypes:
    BOOL = lambda modifier=None: DataType("bool", "(t|f)", modifier)
    # SQLite max is 18446744073709551615
    # This is 20 digits long, so to be safe, cap the generation at 18
    INT = lambda modifier=None: DataType("int", "(\d{1,18})", modifier)
    FLOAT = lambda modifier=None: DataType("float", "(\d(\d|\.)*)", modifier)
    STR = lambda modifier=None: DataType("str", None, modifier)
    LIST = lambda modifier="*": DataType("list", None, modifier)
    ANY = lambda modifier=None: DataType("Any", None, modifier)


STR_TO_DATATYPE: Dict[str, DataType] = {
    "str": DataTypes.STR(),
    "int": DataTypes.INT(),
    "float": DataTypes.FLOAT(),
    "bool": DataTypes.BOOL(),
    "List[str]": DataTypes.STR("*"),
    "List[int]": DataTypes.INT("*"),
    "List[float]": DataTypes.FLOAT("*"),
    "List[bool]": DataTypes.BOOL("*"),
}
