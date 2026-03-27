import asyncio
import datetime
import types
import uuid
import pandas as pd

from blendsql.common.typing import DataType

_PYTHON_TYPE_TO_STR: dict[type, str] = {
    int: "int",
    float: "float",
    str: "str",
    bool: "bool",
    datetime.date: "date",
}

_LIST_ELEM_TYPE_TO_STR: dict[type, str] = {
    int: "list[int]",
    float: "list[float]",
    str: "list[str]",
    bool: "list[bool]",
    datetime.date: "list[date]",
}


def _resolve_return_type(return_type) -> str | DataType | None:
    if return_type is None or isinstance(return_type, (str, DataType)):
        return return_type
    if isinstance(return_type, types.GenericAlias):
        # e.g. list[int], list[str]
        if return_type.__origin__ is list and return_type.__args__:
            return _LIST_ELEM_TYPE_TO_STR.get(return_type.__args__[0], "list[str]")
    if isinstance(return_type, type):
        return _PYTHON_TYPE_TO_STR.get(return_type, "str")
    return "str"


@pd.api.extensions.register_dataframe_accessor("llmmap")
class _LLMMapAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(
        self,
        question: str,
        context_cols: list[str],
        return_type=None,
        **kwargs,
    ):
        import blendsql.configure as _configure
        from blendsql.ingredients.builtin.map.main import LLMMap
        from blendsql.common.typing import AdditionalMapArg

        model = _configure._default_model
        df = self._obj

        for c in context_cols:
            if c not in df.columns:
                raise ValueError(f"Column {c} not in dataframe")

        if len(context_cols) < 1:
            raise ValueError(
                f"The `.llmmap()` API needs at least one column name passed to `context`."
            )

        primary_col = context_cols[0]
        values = df[primary_col].tolist()

        additional_args = []
        for col in context_cols[1:]:
            arg = AdditionalMapArg(columnname=col, tablename="pandas")
            arg.values = df[col].tolist()
            additional_args.append(arg)

        ingredient = LLMMap(name="llmmap", db=None, session_uuid=uuid.uuid4().hex)

        return asyncio.run(
            ingredient.run(
                model=model,
                question=question,
                values=values,
                additional_args=additional_args,
                list_options_in_prompt=ingredient.list_options_in_prompt,
                context_formatter=ingredient.context_formatter,
                return_type=_resolve_return_type(return_type),
                tablename="pandas",
                colname=primary_col,
                **kwargs,
            )
        )
