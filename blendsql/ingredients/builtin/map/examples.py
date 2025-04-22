from attr import attrs, attrib, Factory
import typing as t
from textwrap import dedent

from blendsql.ingredients.few_shot import Example
from blendsql.types import DataType, DataTypes, STR_TO_DATATYPE


@attrs(kw_only=True)
class _MapExample(Example):
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
            # f() returns the output to the question '{self.question}'
            """
        )
        return s


@attrs(kw_only=True)
class MapExample(_MapExample):
    values: t.List[str] = attrib(default=None)


@attrs(kw_only=True)
class AnnotatedMapExample(_MapExample):
    mapping: t.Dict[str, str] = attrib(default=Factory(dict))
