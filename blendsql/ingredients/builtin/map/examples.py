from attr import attrs, attrib
import typing as t
from enum import Enum

from blendsql.ingredients.few_shot import Example
from blendsql.types import DataType, DataTypes, STR_TO_DATATYPE
from blendsql.common.constants import DEFAULT_ANS_SEP, INDENT


class ContextType(Enum):
    GLOBAL = "global"
    LOCAL = "local"


@attrs(kw_only=True)
class MapExample(Example):
    question: str = attrib(default=None)
    context: t.Optional[t.Union[str, t.List[str]]] = attrib(default=None)
    table_name: str = attrib(default=None)
    column_name: str = attrib(default=None)
    options: t.Optional[t.Collection[str]] = attrib(default=None)
    example_outputs: t.Optional[t.List[str]] = attrib(default=None)
    mapping: t.Optional[t.Dict[str, str]] = attrib(default=None)
    return_type: DataType = attrib(
        converter=lambda s: STR_TO_DATATYPE[s] if isinstance(s, str) else s,
        default=DataTypes.ANY(),
    )
    context_type: ContextType = attrib(default=None)


@attrs(kw_only=True)
class AnnotatedMapExample(MapExample):
    mapping: t.Dict[str, str] = attrib()


# Below are for use with constrained models
class ConstrainedMapExample(MapExample):
    def to_string(
        self,
        list_options: bool = True,
        add_leading_newlines: bool = True,
        *args,
        **kwargs,
    ) -> str:
        use_context = self.context_type is not None

        s = "\n\n" if add_leading_newlines else ""
        if list_options and self.options is not None:
            type_annotation = (
                f"t.Literal["
                + ", ".join([f'"{option}"' for option in self.options])
                + "]"
            )
        else:
            type_annotation = self.return_type.name

        if self.table_name and self.column_name:
            args_str = f'Value from the "{self.table_name}"."{self.column_name}" column in a SQL database.'
        else:
            args_str = "Value from a column in a SQL database."

        # Create function signature
        if self.context_type == ContextType.LOCAL:
            s += f"""\ndef f(s: str, context: List[str])"""
        else:
            s += f"""\ndef f(s: str)"""
        s += f' -> {type_annotation}:\n{INDENT()}"""{self.question}'
        if self.context_type == ContextType.GLOBAL:
            s += (
                f"""\n{INDENT()}All function outputs are based on the following context:\n{INDENT()}"""
                + f"\n{INDENT()}{self.context}"
            )

        s += f"""\n\n{INDENT()}Args:\n{INDENT(2)}s (str): {args_str}"""
        if self.context_type == ContextType.LOCAL:
            s += f"""\n{INDENT(2)}context (List[str]): Context to use in answering the question."""
        s += f"""\n\n{INDENT()}Returns:\n{INDENT(2)}{self.return_type.name}: Answer to the above question for each value `s`."""
        s += f"""\n\n{INDENT()}Examples:\n{INDENT(2)}```python"""
        s += (
            f"\n{INDENT(2)}# f() returns the output to the question '{self.question}'"
            + ("" if not use_context else f" given the supplied context")
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
        s += f'''\n{INDENT(2)}```\n{INDENT()}"""\n{INDENT()}...'''
        return s


# Below are for use with unconstrained models
class UnconstrainedMapExample(MapExample):
    def to_string(
        self, values: t.List[str] = None, list_options: bool = True, *args, **kwargs
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
        if self.context_type == ContextType.GLOBAL:
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
