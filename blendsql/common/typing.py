from enum import Enum, EnumMeta
from dataclasses import dataclass
from typing import Literal, Callable, Any


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
QuantifierType = Literal["*", "+"] | str | None


@dataclass
class DataType:
    _name: str
    regex: str | None
    quantifier: QuantifierType | None
    _coerce_fn: Callable

    @property
    def name(self) -> str:
        if self._name != "list" and self.quantifier is not None:
            return f"List[{self._name}]"
        return self._name

    def coerce_fn(self, s: str | None) -> Any:
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
