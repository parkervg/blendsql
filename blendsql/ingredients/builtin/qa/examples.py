from attr import attrs, attrib, validators
import pandas as pd
from typing import Optional, List, Callable

from blendsql.ingredients.few_shot import Example


@attrs(kw_only=True)
class QAExample(Example):
    question: str = attrib()
    context: pd.DataFrame = attrib(
        converter=lambda d: pd.DataFrame.from_dict(d) if isinstance(d, dict) else d,
        validator=validators.instance_of(pd.DataFrame),
    )
    options: Optional[List[str]] = attrib(default=None)

    def to_string(self, context_formatter: Callable[[pd.DataFrame], str]) -> str:
        s = ""
        s += f"\n\nQuestion: {self.question}\n"
        if self.options is not None:
            s += f"Options: {', '.join(self.options)}\n"
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
