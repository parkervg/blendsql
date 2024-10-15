from attr import attrs, attrib, Factory
from typing import Optional, List, Literal, Dict

from blendsql.ingredients.few_shot import Example


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
    mapping: Dict[str, str] = None

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
                values = self.mapping.keys()
            for _idx, k in enumerate(values):
                s += f"{k}\n"
        s += "\nAnswer: "
        return s


@attrs(kw_only=True)
class MapExample(_MapExample):
    values: List[str] = attrib()


@attrs(kw_only=True)
class AnnotatedMapExample(_MapExample):
    mapping: Dict[str, str] = attrib(default=Factory(dict))
