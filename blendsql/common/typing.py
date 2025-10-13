from enum import Enum, EnumMeta
from dataclasses import dataclass
import typing as t


class StrInMeta(EnumMeta):
    def __contains__(cls, item):
        return item in cls.__members__.values()


class IngredientType(str, Enum, metaclass=StrInMeta):
    MAP = "MAP"
    STRING = "STRING"
    QA = "QA"
    JOIN = "JOIN"
    ALIAS = "ALIAS"


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


class IngredientArgType(str):
    pass


class Subquery(IngredientArgType):
    pass


class ColumnRef(IngredientArgType):
    # '{table}.{column}' syntax
    pass
