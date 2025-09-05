import typing as t
from functools import partialmethod
import guidance
import re
from colorama import Fore

from blendsql.common.logger import logger
from ..types import QuantifierType
from .few_shot import Example


def initialize_retriever(
    examples: t.List[Example],
    num_few_shot_examples: t.Optional[int] = None,
) -> t.Callable[[str], t.List[Example]]:
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


@guidance(stateless=True, dedent=False)
def gen_list(
    lm,
    force_quotes: bool,
    quantifier=None,
    options: t.Optional[t.List[str]] = None,
    regex: t.Optional[str] = None,
):
    if options:
        single_item = guidance.select(options, list_append=True, name="response")
    else:
        single_item = guidance.gen(
            max_tokens=100,
            # If not regex is passed, default to all characters except these specific to list-syntax
            # regex=regex or "[^],'\[\]]*",
            # regex="(?:[],'\[\]]|\\)*",
            # regex="([^'\[\]]|\\')*",
            # Default to all characters, except list-syntax
            regex=regex or "([^\]]|\\')*",
            stop_regex="(\n|',)" if not regex else None,
            list_append=True,
            name="response",
        )  # type: ignore
    quote = "'"
    if not force_quotes:
        quote = guidance.optional(quote)  # type: ignore
    single_item = quote + single_item + quote
    single_item += guidance.optional(", ")  # type: ignore
    return lm + "[" + get_quantifier_wrapper(quantifier)(single_item) + "]"


def get_quantifier_wrapper(
    quantifier: QuantifierType,
) -> t.Callable[[guidance.models.Model], guidance.models.Model]:
    quantifier_wrapper = lambda x: x
    if quantifier is not None:
        if quantifier == "*":
            quantifier_wrapper = guidance.zero_or_more
        elif quantifier == "+":
            quantifier_wrapper = guidance.one_or_more
        elif re.match(r"{\d+(,\d+)?}", quantifier):
            repeats = [
                int(i) for i in quantifier.replace("}", "").replace("{", "").split(",")
            ]
            if len(repeats) == 1:
                repeats = repeats * 2
            min_length, max_length = repeats
            quantifier_wrapper = lambda f: guidance.sequence(
                f, min_length=min_length, max_length=max_length
            )  # type: ignore
        else:
            logger.debug(
                Fore.RED + f"Unable to parse quantifier '{quantifier}'" + Fore.RESET
            )
    return quantifier_wrapper  # type: ignore
