from attr import attrs, attrib
import pandas as pd
from typing import Optional, Callable
from collections.abc import Collection

from blendsql.ingredients.few_shot import Example
from blendsql._constants import DataType, STR_TO_DATATYPE


@attrs(kw_only=True)
class QAExample(Example):
    question: str = attrib()
    context: Optional[pd.DataFrame] = attrib(
        converter=lambda d: pd.DataFrame.from_dict(d) if isinstance(d, dict) else d,
        default=None,
    )
    options: Optional[Collection[str]] = attrib(default=None)
    output_type: DataType = attrib(
        converter=lambda s: STR_TO_DATATYPE[s] if isinstance(s, str) else s,
        default=None,
    )

    def to_string(
        self,
        context_formatter: Callable[[pd.DataFrame], str],
        list_options: bool = True,
        *args,
        **kwargs,
    ) -> str:
        s = f"Question: {self.question}\n"
        if self.output_type is not None:
            if self.output_type.name != "Any":
                s += f"Output datatype: {self.output_type.name}\n"
        if list_options:
            if self.options is not None:
                s += f"Options: {', '.join(sorted(self.options))}\n"
        if self.output_type is not None:
            modifier = self.output_type.modifier
            if modifier is not None:
                if modifier == "*":
                    s += "You may generate zero or more responses in your list.\n"
                elif modifier == "+":
                    s += "You may generate one or more responses in your list.\n"
                else:
                    repeats = [
                        int(i)
                        for i in modifier.replace("}", "").replace("{", "").split(",")
                    ]
                    if len(repeats) == 1:
                        repeats = repeats * 2
                    min_length, max_length = repeats
                    if min_length == max_length:
                        s += f"You may generate {min_length} responses in your list.\n"
                    else:
                        s += f"You may generate between {min_length} and {max_length} responses in your list.\n"
        if self.context is not None:
            s += f"Context:\n{context_formatter(self.context)}"
        s += "\nAnswer: "
        return s


@attrs(kw_only=True)
class AnnotatedQAExample(QAExample):
    answer: str = attrib()

    def to_string(
        self,
        context_formatter: Callable[[pd.DataFrame], str],
        include_answer: bool = False,
    ):
        s = super().to_string(context_formatter=context_formatter)
        if include_answer:
            s += self.answer
        return s
