from attr import attrs, attrib, Factory
import typing as t
from textwrap import dedent

from blendsql.ingredients.few_shot import Example
from blendsql.type_constraints import DataType, DataTypes, STR_TO_DATATYPE


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
        include_values: bool = True,
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

        s += dedent(
            f"""
        def f(s: str) -> {type_annotation}:
            \"\"\"{self.question}
                        
            Returns:
                {self.output_type.name}: Answer to the above question for the value `s`."""
        )
        # s += '\n\t"""\n\t...\n\n'
        # if self.table_name is not None:
        #     s += f"Source table: {self.table_name}\n"
        # if self.column_name is not None:
        #     s += f"Source column: {self.column_name}\n"
        if include_values:
            s += "\nValues:\n"
            values = self.values
            if self.values is None:
                values = self.mapping.keys()
            for _idx, k in enumerate(values):
                s += f"{k}\n"
        return s


@attrs(kw_only=True)
class MapExample(_MapExample):
    values: t.List[str] = attrib(default=None)


@attrs(kw_only=True)
class AnnotatedMapExample(_MapExample):
    mapping: t.Dict[str, str] = attrib(default=Factory(dict))
