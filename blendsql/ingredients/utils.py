import typing as t
from functools import partialmethod

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
