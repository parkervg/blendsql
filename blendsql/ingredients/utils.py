from typing import Callable
from functools import partialmethod
import guidance
import re

from blendsql.common.logger import logger, Color
from .few_shot import Example

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
        return guidance.select(
            [
                "'" + item + "'",
                '"' + item + '"',
            ]
        )
    else:
        return guidance.select(
            [
                "'" + item + "'",
                '"' + item + '"',
                item,  # unquoted option
            ]
        )


@guidance(stateless=True, dedent=False)
def gen_list(
    lm,
    force_quotes: bool,
    quantifier=None,
    options: list[str] | None = None,
    regex: str | None = None,
):
    if options:
        single_item = guidance.select(options, list_append=False)
    else:
        single_item = guidance.gen(
            max_tokens=100,
            regex=regex,
            # Stop at Python list item separators
            stop_regex=LIST_ITEM_STOP_REGEX if not regex else None,
            list_append=False,
        )  # type: ignore
    quoted_item = _wrap_with_quotes(single_item, bool(options or regex), force_quotes)
    quantifier_fn = get_quantifier_wrapper(quantifier)
    # For quantifiers that allow zero items, we need to handle empty list case
    if quantifier in (None, "*") or (quantifier and quantifier.startswith("{0")):
        # Could be empty, so entire contents is optional
        subsequent_items = guidance.zero_or_more(", " + quoted_item)
        list_contents = guidance.optional(quoted_item + subsequent_items)
    elif quantifier == "+":
        # One or more: first item + zero or more (sep + item)
        subsequent_items = guidance.zero_or_more(", " + quoted_item)
        list_contents = quoted_item + subsequent_items
    elif quantifier and re.match(r"{\d+(,\d*)?}", quantifier):
        # Specific count - use the quantifier wrapper but adjust for separator
        item_with_sep = quoted_item + guidance.optional(", ")
        list_contents = quantifier_fn(item_with_sep)
    else:
        # Default: single item
        list_contents = quoted_item

    return lm + "[" + list_contents + "]"


def get_quantifier_wrapper(
    quantifier: str | None,
) -> Callable[[guidance.models.Model], guidance.models.Model]:
    """
    Returns a wrapper function that applies the appropriate quantifier
    to a grammar element.
    """
    quantifier_wrapper = lambda x: x

    if quantifier is not None:
        if quantifier == "*":
            quantifier_wrapper = guidance.zero_or_more
        elif quantifier == "+":
            quantifier_wrapper = guidance.one_or_more
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

            quantifier_wrapper = (
                lambda f, mn=min_length, mx=max_length: guidance.sequence(
                    f, min_length=mn, max_length=mx
                )
            )
        else:
            logger.debug(Color.error(f"Unable to parse quantifier '{quantifier}'"))

    return quantifier_wrapper
