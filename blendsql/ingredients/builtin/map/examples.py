from attr import attrs, attrib
import typing as t

from blendsql.ingredients.few_shot import Example
from blendsql.types import DataType, DataTypes, STR_TO_DATATYPE


@attrs(kw_only=True)
class MapExample(Example):
    question: str = attrib(default=None)
    context: t.Optional[str] = attrib(default=None)
    values: t.Optional[t.List[str]] = attrib(default=None)
    table_name: str = attrib(default=None)
    column_name: str = attrib(default=None)
    options: t.Optional[t.Collection[str]] = attrib(default=None)
    example_outputs: t.Optional[t.List[str]] = attrib(default=None)
    mapping: t.Optional[t.Dict[str, str]] = attrib(default=None)
    return_type: DataType = attrib(
        converter=lambda s: STR_TO_DATATYPE[s] if isinstance(s, str) else s,
        default=DataTypes.STR(),
    )
    use_context: t.Optional[bool] = attrib(default=False)


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
            type_annotation = self.return_type.name

        if self.table_name and self.column_name:
            args_str = f'Value from the "{self.table_name}"."{self.column_name}" column in a SQL database.'
        else:
            args_str = "Value from a column in a SQL database."

        s += (
            f"""\ndef f(s: str"""
            + (", context: str" if self.use_context else "")
            + f') -> {type_annotation}:\n\t"""{self.question}'
        )
        s += f"""\n\n\tArgs:\n\t\ts (str): {args_str}"""
        if self.use_context:
            s += f"""\n\t\tcontext (str): Context to use in answering the question."""
        s += f"""\n\n\tReturns:\n\t\t{self.return_type.name}: Answer to the above question for each value `s`."""
        s += """\n\n\tExamples:\n\t\t```python"""
        s += f"\n\t\t# f() returns the output to the question '{self.question}'" + (
            "" if self.context is None else f" given the supplied context"
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
