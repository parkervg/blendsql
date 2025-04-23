from attr import attrs, attrib
import typing as t
from textwrap import dedent

from blendsql.ingredients.few_shot import Example
from blendsql.types import DataType, DataTypes, STR_TO_DATATYPE
from blendsql.common.constants import DEFAULT_ANS_SEP


@attrs(kw_only=True)
class MapExample(Example):
    question: str = attrib()
    table_name: str = attrib(default=None)
    column_name: str = attrib(default=None)
    options: t.Optional[t.Collection[str]] = attrib(default=None)
    example_outputs: t.Optional[t.List[str]] = attrib(default=None)
    values: t.Optional[t.List[str]] = attrib(default=None)
    mapping: t.Optional[t.Dict[str, str]] = attrib(default=None)
    output_type: DataType = attrib(
        converter=lambda s: STR_TO_DATATYPE[s] if isinstance(s, str) else s,
        default=DataTypes.STR(),
    )


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
        s = "\n\n" if add_leading_newlines else ""
        if list_options and self.options is not None:
            type_annotation = (
                f"t.Literal["
                + ", ".join([f'"{option}"' for option in self.options])
                + "]"
            )
        else:
            type_annotation = self.output_type.name

        if self.table_name and self.column_name:
            args_str = f'Value from the "{self.table_name}"."{self.column_name}" column in a SQL database.'
        else:
            args_str = "Value from a column in a SQL database."

        s += dedent(
            f"""
        def f(s: str) -> {type_annotation}:
            \"\"\"{self.question}

            Args:
                s (str): {args_str}

            Returns:
                {self.output_type.name}: Answer to the above question for each value `s`.

            Examples:
                ```python
                # f() returns the output to the question '{self.question}'"""
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
            s += f'\n\t\tf("{k}") == ' + (f'"{v}"' if isinstance(v, str) else f"{v}")
        s += '''\n\t\t```\n\t"""\n\t...'''
        return s


# Below are for use with unconstrained models
class UnconstrainedMapExample(MapExample):
    def to_string(self, list_options: bool = True, *args, **kwargs) -> str:
        s = f"\n\nQuestion: {self.question}\n"
        if self.table_name is not None:
            s += f"Source table: {self.table_name}\n"
        if self.column_name is not None:
            s += f"Source column: {self.column_name}\n"
        if self.output_type is not None:
            if self.output_type.name != "Any":
                s += f"Output datatype: {self.output_type.name}\n"
        # if self.example_outputs is not None:
        #     s += f"Example outputs: {';'.join(self.example_outputs)}\n"
        if list_options:
            if self.options is not None:
                s += f"Options: {','.join(sorted(self.options))}\n"
        s += "\nValues:\n"
        values = self.values
        if self.values is None:
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
