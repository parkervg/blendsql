from enum import Enum, EnumMeta
from dataclasses import dataclass, field
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
    atomic_type: str
    regex: str | None
    quantifier: QuantifierType | None
    _coerce_fn: Callable
    requires_quotes: bool = False

    @property
    def name(self) -> str:
        if self.atomic_type != "list" and self.quantifier is not None:
            return f"List[{self.atomic_type}]"
        return self.atomic_type

    def coerce_fn(self, s: str | None, db: "Database | None") -> Any:
        """Language models output strings.
        We want to coerce their outputs to the DB-friendly type here.
        """
        return self._coerce_fn(s, db)


class IngredientArgType:
    pass


class Subquery(IngredientArgType, str):
    pass


class ColumnRef(IngredientArgType, str):
    # '{table}.{column}' syntax
    pass


@dataclass
class AdditionalMapArg:
    values: list[Any] = field(init=False)
    columnname: str
    tablename: str


class StringConcatenation(IngredientArgType, list):
    # Some type of `column1 || ' ' || column2` expression

    def __init__(self, columns: list[ColumnRef], raw_expr: str):
        self.raw_expr = raw_expr
        super().__init__(columns)

    def __hash__(self):
        return hash(tuple(self))

    def __eq__(self, other):
        if isinstance(other, StringConcatenation):
            return list.__eq__(self, other)
        return NotImplemented
