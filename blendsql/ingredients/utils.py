from typing import Callable
from functools import partialmethod
from guidance._grammar import select
from guidance.library._sequences import one_or_more, zero_or_more, sequence
from guidance import json as guidance_json
from typing import Literal

import re

from pydantic import TypeAdapter

from blendsql.common.logger import logger, Color
from .few_shot import Example
from ..common.typing import DataType

LIST_ITEM_STOP_REGEX = r"(\n|',|\",|'\]|\"\])"


def initialize_retriever(
    examples: list[Example],
    num_few_shot_examples: int | None = None,
) -> Callable[[str], list[Example]]:
    """Initializes a DPR retriever over the few-shot examples provided."""
    if num_few_shot_examples == 0:
        return lambda *_: []
    else:
        return lambda *_: examples[:num_few_shot_examples]


def partialclass(cls, *args, **kwds):
    # https://stackoverflow.com/a/38911383
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds)

    NewCls.__name__ = cls.__name__
    return NewCls


def _wrap_with_quotes(item, has_options_or_regex: bool, force_quotes: bool):
    if not force_quotes and has_options_or_regex:
        # Don't add quotes on e.g. boolean lists
        return item
    elif force_quotes:
        return select(
            [
                "'" + item + "'",
                '"' + item + '"',
            ]
        )
    else:
        return select(
            [
                "'" + item + "'",
                '"' + item + '"',
                item,  # unquoted option
            ]
        )


def get_python_type(
    data_type: DataType,
    options: list[str] | None = None,
):
    if options:
        return Literal[tuple(options)]
    else:
        item_type = data_type.atomic_type
    return item_type


def gen_list(
    data_type: DataType,
    quantifier=None,
    options: list[str] | None = None,
):
    item_type = get_python_type(
        data_type=data_type,
        options=options,
    )
    if quantifier is None:
        # Default: single item in a list
        schema = TypeAdapter(tuple[item_type])
    elif quantifier == "+":
        # One or more
        schema = TypeAdapter(
            list[item_type]
        )  # minItems=1 handled downstream or via annotation
    elif quantifier == "*":
        schema = TypeAdapter(list[item_type])
    elif match := re.match(r"\{(\d+)\}", quantifier):
        count = int(match.group(1))
        # Fixed-length tuple: tuple[item_type, item_type, ...] with `count` elements
        schema = TypeAdapter(tuple[tuple(item_type for _ in range(count))])
    elif match := re.match(r"\{(\d+),(\d*)\}", quantifier):
        min_count = int(match.group(1))
        max_count = int(match.group(2)) if match.group(2) else None
        from pydantic import Field
        from typing import Annotated

        if max_count is not None:
            schema = TypeAdapter(
                Annotated[
                    list[item_type], Field(min_length=min_count, max_length=max_count)
                ]
            )
        else:
            schema = TypeAdapter(
                Annotated[list[item_type], Field(min_length=min_count)]
            )
    else:
        schema = TypeAdapter(list[item_type])

    return guidance_json(schema=schema)


def get_quantifier_wrapper(
    quantifier: str | None,
) -> Callable:
    """
    Returns a wrapper function that applies the appropriate quantifier
    to a grammar element.
    """
    quantifier_wrapper = lambda x: x

    if quantifier is not None:
        if quantifier == "*":
            quantifier_wrapper = zero_or_more
        elif quantifier == "+":
            quantifier_wrapper = one_or_more
        elif re.match(r"{\d+(,\d*)?}", quantifier):
            inner = quantifier.strip("{}")
            parts = inner.split(",")
            min_length = int(parts[0])

            # Handle all three cases: {n}, {n,m}, and {n,}
            if len(parts) == 1:
                # Exact count: {n}
                max_length = min_length
            elif parts[1] == "":
                # Unbounded upper: {n,}
                max_length = None
            else:
                # Range: {n,m}
                max_length = int(parts[1])

            quantifier_wrapper = lambda f, mn=min_length, mx=max_length: sequence(
                f, min_length=mn, max_length=mx
            )
        else:
            logger.debug(Color.error(f"Unable to parse quantifier '{quantifier}'"))

    return quantifier_wrapper
