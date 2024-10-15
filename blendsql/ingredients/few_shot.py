from typing import List, Callable, Optional, Literal, Dict
import pandas as pd
from attr import attrs, attrib, Factory, validators
from abc import abstractmethod

from blendsql.utils import newline_dedent


class Example:
    @abstractmethod
    def to_string(self, *args, **kwargs) -> str:
        ...


# LLMQA
@attrs(kw_only=True)
class QAExample(Example):
    question: str = attrib()
    context: pd.DataFrame = attrib(validator=validators.instance_of(pd.DataFrame))
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


# LLMMap
@attrs(kw_only=True)
class _MapExample(Example):
    question: str = attrib()
    table_name: str = attrib(default=None)
    column_name: str = attrib(default=None)
    output_type: Optional[Literal["boolean", "integer", "float", "string"]] = attrib(
        default=None
    )
    options: Optional[List[str]] = attrib(default=None)
    example_outputs: Optional[List[str]] = attrib(default=None)

    values: List[str] = None
    examples: Dict[str, str] = None

    def to_string(self, include_values: bool = True) -> str:
        s = "\n\n"
        s += f"\n\nQuestion: {self.question}\n"
        if self.table_name is not None:
            s += f"Source table: {self.table_name}\n"
        if self.column_name is not None:
            s += f"Source column: {self.column_name}\n"
        if self.output_type is not None:
            s += f"Output type: {self.output_type}\n"
        if self.example_outputs is not None:
            s += f"Example outputs: {';'.join(self.example_outputs)}\n"
        if self.options is not None:
            s += f"Options: {','.join(self.options)}"
            s += "\n"
        if include_values:
            s += "\nValues:\n"
            values = self.values
            if self.values is None:
                values = self.examples.keys()
            for _idx, k in enumerate(values):
                s += f"{k}\n"
        s += "\nAnswer: "
        return s


@attrs(kw_only=True)
class MapExample(_MapExample):
    values: List[str] = attrib()


@attrs(kw_only=True)
class AnnotatedMapExample(_MapExample):
    examples: Dict[str, str] = attrib(default=Factory(dict))


# LLMJoin
@attrs(kw_only=True)
class JoinExample(Example):
    join_criteria: str = attrib(default="Join to the same topics.")
    left_values: List[str] = attrib()
    right_values: List[str] = attrib()

    def to_string(self) -> str:
        return newline_dedent(
            """
        Criteria: {}

        Left Values:
        {}

        Right Values:
        {}

        Output:
        """.format(
                self.join_criteria,
                "\n".join(self.left_values),
                "\n".join(self.right_values),
            )
        )


@attrs(kw_only=True)
class AnnotatedJoinExample(JoinExample):
    mapping: Dict[str, str] = attrib()
