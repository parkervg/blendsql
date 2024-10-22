from typing import Union, List, Set, Dict, Callable
from functools import partialmethod, partial
from ast import literal_eval
from colorama import Fore

from ..utils import get_tablename_colname
from ..db import Database
from .few_shot import Example
from blendsql._constants import DEFAULT_NAN_ANS
from .._logger import logger


def unpack_options(
    options: Union[List[str], str], aliases_to_tablenames: Dict[str, str], db: Database
) -> Set[str]:
    unpacked_options = options
    if not isinstance(options, list):
        try:
            tablename, colname = get_tablename_colname(options)
            tablename = aliases_to_tablenames.get(tablename, tablename)
            # Optionally materialize a CTE
            if tablename in db.lazy_tables:
                unpacked_options: list = (
                    db.lazy_tables.pop(tablename).collect()[colname].unique().tolist()
                )
            else:
                unpacked_options: list = db.execute_to_list(
                    f'SELECT DISTINCT "{colname}" FROM "{tablename}"'
                )
        except ValueError:
            unpacked_options = options.split(";")
    return set(unpacked_options)


def initialize_retriever(
    examples: Example, k: int = None, **to_string_args
) -> Callable[[str], List[Example]]:
    """Initializes a DPR retriever over the few-shot examples provided."""
    if k is None or k == len(examples):
        # Just return all the examples everytime this is called
        return lambda *_: examples
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


def cast_responses_to_datatypes(responses: List[str]) -> List[Union[float, int, str]]:
    responses = [
        {
            "t": True,
            "f": False,
            "true": True,
            "false": False,
            "y": True,
            "n": False,
            "yes": True,
            "no": False,
            DEFAULT_NAN_ANS: None,
        }.get(i.lower(), i)
        if isinstance(i, str)
        else i
        for i in responses
    ]
    # Try to cast strings as numerics
    for idx, value in enumerate(responses):
        if not isinstance(value, str):
            continue
        value = value.replace(",", "")
        try:
            casted_value = literal_eval(value)
            assert isinstance(casted_value, (float, int, str))
            responses[idx] = casted_value
        except (ValueError, SyntaxError, AssertionError):
            continue
    return responses


def partialclass(cls, *args, **kwds):
    # https://stackoverflow.com/a/38911383
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds)

    NewCls.__name__ = cls.__name__
    return NewCls
