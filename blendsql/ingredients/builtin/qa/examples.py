from attr import attrs, attrib
import pandas as pd
import typing as t
from collections.abc import Collection

from blendsql.ingredients.few_shot import Example
from blendsql.types import DataType, DataTypes, STR_TO_DATATYPE


@attrs(kw_only=True)
class QAExample(Example):
    question: str = attrib()
    context: t.List[pd.DataFrame] = attrib(
        converter=lambda d: [pd.DataFrame.from_dict(d)] if isinstance(d, dict) else d,
        default=None,
    )
    options: t.Optional[Collection[str]] = attrib(default=None)
    return_type: DataType = attrib(
        converter=lambda s: STR_TO_DATATYPE[s] if isinstance(s, str) else s,
        default=DataTypes.ANY(),
    )

    def to_string(
        self,
        context_formatter: t.Callable[[pd.DataFrame], str],
        list_options: bool = True,
        *args,
        **kwargs,
    ) -> str:
        s = f"Question: {self.question}\n"
        if self.return_type is not None:
            if self.return_type._name not in {"Any"}:
                s += f"Output datatype: {self.return_type.name}\n"
        if list_options:
            if self.options is not None:
                # s += f"Options: {', '.join(sorted(self.options))}\n"
                s += f"Options: {list(sorted(self.options))}\n"
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


@attrs(kw_only=True)
class AnnotatedQAExample(QAExample):
    answer: str = attrib()

    def to_string(
        self,
        context_formatter: t.Callable[[pd.DataFrame], str],
        include_answer: bool = False,
    ):
        s = super().to_string(context_formatter=context_formatter)
        if include_answer:
            s += self.answer
        return s
