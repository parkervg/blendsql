import typing as t
from functools import partialmethod

from .few_shot import Example

# Can be:
# - '{value1};{value2}...' syntax
# - '{table}::{column}' syntax
# - A BlendSQL query which returns a 1d array of values ((SELECT value FROM table WHERE ...))
ValueArray = t.NewType("ValueArray", str)


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
    from blendsql.vector_stores.faiss_vector_store import FaissVectorStore

    retriever = FaissVectorStore(
        documents=[example.to_string(**to_string_args) for example in examples],
        return_objs=examples,
        k=k,
    )
    return retriever


def partialclass(cls, *args, **kwds):
    # https://stackoverflow.com/a/38911383
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds)

    NewCls.__name__ = cls.__name__
    return NewCls
