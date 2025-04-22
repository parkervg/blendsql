import typing as t
from functools import partialmethod, partial
from colorama import Fore

from .few_shot import Example
from blendsql.common.logger import logger


def initialize_retriever(
    examples: t.List[Example], k: t.Optional[int] = None, **to_string_args
) -> t.Callable[[str], t.List[Example]]:
    """Initializes a DPR retriever over the few-shot examples provided."""
    if k is None or k == len(examples):
        # Just return all the examples everytime this is called
        return lambda *_: examples
    elif k == 0:
        return lambda *_: []

    assert k <= len(
        examples
    ), f"The `k` argument to an ingredient must be less than `len(few_shot_examples)`!\n`k` is {k}, `len(few_shot_examples)` is {len(examples)}"
    from .retriever import Retriever

    logger.debug(Fore.YELLOW + "Processing documents with haystack..." + Fore.RESET)
    retriever = Retriever(
        documents=[example.to_string(**to_string_args) for example in examples],
        return_objs=examples,
    )
    return partial(retriever.retrieve_top_k, k=k)


def partialclass(cls, *args, **kwds):
    # https://stackoverflow.com/a/38911383
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds)

    NewCls.__name__ = cls.__name__
    return NewCls
