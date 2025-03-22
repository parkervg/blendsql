from typing import Union, List, Set, Dict, Callable, Optional, Union
from collections.abc import Collection
from functools import partialmethod, partial
from ast import literal_eval
from colorama import Fore

from ..utils import get_tablename_colname
from ..db import Database
from .few_shot import Example
from .._logger import logger
from blendsql._exceptions import IngredientException
from blendsql._constants import (
    DEFAULT_NAN_ANS,
    ModifierType,
    DataType,
    DataTypes,
    STR_TO_DATATYPE,
)


def unpack_options(
    options: Union[List[str], str], aliases_to_tablenames: Dict[str, str], db: Database
) -> Union[Set[str], None]:
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
    return set(unpacked_options) if len(unpacked_options) > 0 else None


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


def prepare_datatype(
    options: Optional[Collection[str]],
    output_type: Optional[Union[str]] = None,
    modifier: Optional[ModifierType] = None,
) -> DataType:
    if output_type is None:
        output_type = DataTypes.ANY()
    elif isinstance(output_type, str):
        # The user has passed us an output type in the BlendSQL query
        # That should take precedence
        if output_type not in STR_TO_DATATYPE:
            raise IngredientException(
                f"{output_type} is not a recognized datatype!\nValid options are {list(STR_TO_DATATYPE.keys())}"
            )
        output_type = STR_TO_DATATYPE.get(output_type)
        if modifier:  # User passed modifier takes precedence
            output_type.modifier = modifier
    if modifier:
        # The user has passed us a modifier that should take precedence
        output_type.modifier = modifier
    if output_type.regex is not None:
        if options is not None:
            logger.debug(
                Fore.LIGHTBLACK_EX
                + f"Ignoring inferred regex '{output_type.regex}' and using options '{options}' instead"
                + Fore.RESET
            )
            output_type.regex = None
        else:
            logger.debug(
                Fore.LIGHTBLACK_EX + f"Using regex '{output_type.regex}'" + Fore.RESET
            )
    elif options:
        logger.debug(Fore.LIGHTBLACK_EX + f"Using options '{options}'" + Fore.RESET)
    return output_type


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
