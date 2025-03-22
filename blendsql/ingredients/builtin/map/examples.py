from attr import attrs, attrib, Factory
from typing import Optional, List, Dict
from collections.abc import Collection

from blendsql.ingredients.few_shot import Example
from blendsql._constants import DataType, STR_TO_DATATYPE


@attrs(kw_only=True)
class _MapExample(Example):
    question: str = attrib()
    table_name: str = attrib(default=None)
    column_name: str = attrib(default=None)
    output_type: DataType = attrib(
        converter=lambda s: STR_TO_DATATYPE[s] if isinstance(s, str) else s,
        default=None,
    )
    options: Optional[Collection[str]] = attrib(default=None)
    example_outputs: Optional[List[str]] = attrib(default=None)

    values: List[str] = None
    mapping: Dict[str, str] = None

    def to_string(
        self, include_values: bool = True, list_options: bool = True, *args, **kwargs
    ) -> str:
        s = f"\n\nQuestion: {self.question}\n"
        if self.table_name is not None:
            s += f"Source table: {self.table_name}\n"
        if self.column_name is not None:
            s += f"Source column: {self.column_name}\n"
        if self.output_type is not None:
            if self.output_type.name != "Any":
                s += f"Output datatype: {self.output_type.name}\n"
        if self.example_outputs is not None:
            s += f"Example outputs: {';'.join(self.example_outputs)}\n"
        if list_options:
            if self.options is not None:
                s += f"Options: {','.join(sorted(self.options))}\n"
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
    values: List[str] = attrib(default=None)


@attrs(kw_only=True)
class AnnotatedMapExample(_MapExample):
    mapping: Dict[str, str] = attrib(default=Factory(dict))
