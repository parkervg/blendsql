from dataclasses import dataclass, field
from collections.abc import Collection
from enum import Enum
from textwrap import indent

from blendsql.ingredients.few_shot import Example
from blendsql.types import DataTypes, STR_TO_DATATYPE
from blendsql.common.typing import DataType, AdditionalMapArg
from blendsql.common.constants import DEFAULT_ANS_SEP, INDENT


def _return_type_converter(value):
    return STR_TO_DATATYPE[value.lower()] if isinstance(value, str) else value


class FeatureType(Enum):
    """Distinguishes between features that are passed for each value (LOCAL),
    vs. ones that can be shared and prefix-cached for an entire inference session (GLOBAL).
    """

    GLOBAL = "global"
    LOCAL = "local"


@dataclass(kw_only=True)
class MapExample(Example):
    question: str = field(default=None)
    context: str | list[str] | None = field(default=None)
    context_type: FeatureType = field(default=None)
    table_name: str = field(default=None)
    column_name: str = field(default=None)
    options: Collection[str] | None = field(default=None)
    options_type: FeatureType = field(default=None)
    example_outputs: list[str] | None = field(default=None)
    mapping: dict[str, str] | None = field(default=None)
    return_type: DataType = field(default_factory=lambda: DataTypes.ANY())

    def __post_init__(self):
        # Apply converters
        self.return_type = _return_type_converter(self.return_type)


@dataclass(kw_only=True)
class AnnotatedMapExample(MapExample):
    mapping: dict[str, str] = field()


# Below are for use with constrained models
class ConstrainedMapExample(MapExample):
    def to_string(
        self,
        list_options: bool = True,
        add_leading_newlines: bool = True,
        use_local_options: bool = False,
        additional_args: list[AdditionalMapArg] = None,
        *args,
        **kwargs,
    ) -> str:
        if additional_args is None:
            additional_args = []

        use_context = self.context_type is not None

        s = "\n\n" if add_leading_newlines else ""
        s += "```python\n"
        if list_options and self.options is not None:
            return_type_annotation = (
                f"Literal["
                + ", ".join(
                    [
                        f'"{option}"'
                        if self.return_type.requires_quotes
                        else str(option)
                        for option in self.options
                    ]
                )
                + "]"
            )
        else:
            return_type_annotation = self.return_type.name

        if self.table_name and self.column_name:
            args_str = f'Values from the "{self.table_name}" table in a SQL database.'
        else:
            args_str = "Value from a column in a SQL database."

        # Create function signature
        var_name = self.column_name or "s"
        s += f"""def f({var_name}: str"""

        for arg in additional_args:
            s += f""", {arg.columnname}: str"""

        if self.context_type == FeatureType.LOCAL:
            s += f""", context: List[str]"""

        if self.options_type == FeatureType.LOCAL:
            s += f""", options: List[str]"""
        s += ")"
        s += f" -> {return_type_annotation}:\n" + indent(
            f'"""{self.question}', prefix=INDENT()
        )

        if self.context_type == FeatureType.GLOBAL:
            indented_context = self.context.replace("\n", "\n" + INDENT())
            s += (
                f"""\n{INDENT()}All function outputs are based on the following context:\n{INDENT()}"""
                + f"\n{INDENT()}{indented_context}"
            )
        arg_name = self.column_name or "s"
        s += f"""\n\n{INDENT()}Args:\n{INDENT(2)}{arg_name} (str): {args_str}"""
        for arg in additional_args:
            s += f"""\n{INDENT(2)}{arg.columnname} (str): Values from the "{arg.tablename}" table in a SQL database."""
        if self.context_type == FeatureType.LOCAL:
            s += f"""\n{INDENT(2)}context (List[str]): Context to use in answering the question."""
        if self.options_type == FeatureType.LOCAL:
            s += f"""\n{INDENT(2)}options (List[str]): Candidate strings for use in your response."""
        s += f"""\n\n{INDENT()}Returns:\n{INDENT(2)}{return_type_annotation}: Answer to the above question for each input."""
        s += f"""\n\n{INDENT()}Examples:\n{INDENT(2)}```python"""
        _question = '"' + self.question + '"'
        if "\n" in self.question:
            _question = "\n" + indent(self.question, prefix=INDENT(2))
            _question = '"""' + _question + INDENT(2) + '"""'
        s += f"\n{INDENT(2)}# f() returns the output to the question {_question}" + (
            "" if not use_context else f" given the supplied context"
        )
        return s


class ConstrainedAnnotatedMapExample(ConstrainedMapExample):
    def to_string(
        self,
        list_options: bool = True,
        add_leading_newlines: bool = True,
        *args,
        **kwargs,
    ) -> str:
        s = super().to_string(
            list_options=list_options,
            add_leading_newlines=add_leading_newlines,
            *args,
            **kwargs,
        )
        for k, v in self.mapping.items():
            s += f'\n{INDENT(2)}f("{k}") == ' + (
                f'"{v}"' if isinstance(v, str) else f"{v}"
            )
        s += f'''\n{INDENT(2)}```\n{INDENT()}"""\n{INDENT()}...\n```'''
        return s


# Below are for use with unconstrained models
class UnconstrainedMapExample(MapExample):
    def to_string(
        self, values: list[str] = None, list_options: bool = True, *args, **kwargs
    ) -> str:
        s = f"\n\nQuestion: {self.question}\n"
        if self.table_name and self.column_name:
            s += f'Source column: "{self.table_name}"."{self.column_name}"\n'
        if self.return_type is not None:
            if self.return_type.name != "Any":
                s += f"Output datatype: {self.return_type.name}\n"
        if list_options:
            if self.options is not None:
                s += f"Options: {','.join(sorted(self.options))}\n"
        if self.context_type == FeatureType.GLOBAL:
            s += f"Context: {self.context}"
        s += "\nValues:\n"
        if values is None:
            values = self.mapping.keys()
        for _idx, k in enumerate(values):
            s += f"{k}\n"
        s += "\nAnswer: "
        return s


class UnconstrainedAnnotatedMapExample(UnconstrainedMapExample):
    def to_string(self, list_options: bool = True, *args, **kwargs) -> str:
        s = super().to_string(list_options=list_options, *args, **kwargs)
        s += DEFAULT_ANS_SEP.join([str(i) for i in self.mapping.values()])
        return s + "\n\n---"
