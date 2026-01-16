from dataclasses import dataclass, field
import polars as pl
from typing import Callable
from collections.abc import Collection

from blendsql.ingredients.few_shot import Example
from blendsql.common.typing import DataType
from blendsql.types import DataTypes, STR_TO_DATATYPE


def _context_converter(value):
    if value is None:
        return None
    return [pl.from_dict(value)] if isinstance(value, dict) else value


def _return_type_converter(value):
    return STR_TO_DATATYPE[value.lower()] if isinstance(value, str) else value


@dataclass(kw_only=True)
class QAExample(Example):
    question: str
    context: list[pl.DataFrame] | None = None
    options: Collection[str] | None = None
    return_type: DataType = field(default_factory=lambda: DataTypes.ANY())

    def __post_init__(self):
        # Apply converters
        self.context = _context_converter(self.context)
        self.return_type = _return_type_converter(self.return_type)

    def to_string(
        self,
        context_formatter: Callable[[pl.DataFrame], str],
        list_options: bool = True,
        *args,
        **kwargs,
    ) -> str:
        s = f"Question: {self.question}\n"
        if self.return_type is not None:
            if self.return_type.atomic_type not in {"Any"}:
                s += f"Output datatype: {self.return_type.name}\n"
        if list_options:
            if self.options is not None:
                s += f"Options: {list(self.options)}\n"
        if self.return_type is not None:
            quantifier = self.return_type.quantifier
            if quantifier is not None:
                if quantifier == "*":
                    s += "You may generate zero or more responses in your list.\n"
                elif quantifier == "+":
                    s += "You may generate one or more responses in your list.\n"
                else:
                    repeats = [
                        int(i)
                        for i in quantifier.replace("}", "").replace("{", "").split(",")
                    ]
                    if len(repeats) == 1:
                        repeats = repeats * 2
                    min_length, max_length = repeats
                    if min_length == max_length:
                        s += f"You may generate {min_length} responses in your list.\n"
                    else:
                        s += f"You may generate between {min_length} and {max_length} responses in your list.\n"
        if self.context is not None:
            s += f"Context:"
            for c in self.context:
                s += f"\n{context_formatter(c)}"
        s += "\nAnswer: "
        return s


@dataclass(kw_only=True)
class AnnotatedQAExample(QAExample):
    answer: str = field(default="")

    def to_string(
        self,
        context_formatter: Callable[[pl.DataFrame], str],
        include_answer: bool = False,
    ):
        s = super().to_string(context_formatter=context_formatter)
        if include_answer:
            s += self.answer
        return s
