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
    if len(unpacked_options) == 0:
        logger.debug(
            Fore.YELLOW
            + f"Tried to unpack options '{options}', but got an empty list\nThis may be a bug. Please report it."
            + Fore.RESET
        )
    return set(unpacked_options) if len(unpacked_options) > 0 else None


def initialize_retriever(
    examples: List[Example], k: Optional[int] = None, **to_string_args
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
    output_type: Optional[Union[str, DataType]] = None,
    modifier: Optional[ModifierType] = None,
) -> DataType:
    if output_type is None:
        resolved_output_type = DataTypes.ANY()
    elif isinstance(output_type, str):
        # The user has passed us an output type in the BlendSQL query
        # That should take precedence
        if output_type not in STR_TO_DATATYPE:
            raise IngredientException(
                f"{output_type} is not a recognized datatype!\nValid options are {list(STR_TO_DATATYPE.keys())}"
            )
        resolved_output_type = STR_TO_DATATYPE[output_type]
        if modifier:  # User passed modifier takes precedence
            resolved_output_type.modifier = modifier
    elif isinstance(output_type, DataType):
        resolved_output_type = output_type
    if modifier:
        # The user has passed us a modifier that should take precedence
        resolved_output_type.modifier = modifier
    if resolved_output_type.regex is not None:
        if options is not None:
            logger.debug(
                Fore.LIGHTBLACK_EX
                + f"Ignoring inferred regex '{resolved_output_type.regex}' and using options '{options}' instead"
                + Fore.RESET
            )
            resolved_output_type.regex = None
        else:
            logger.debug(
                Fore.LIGHTBLACK_EX
                + f"Using regex '{resolved_output_type.regex}'"
                + Fore.RESET
            )
    elif options:
        logger.debug(Fore.LIGHTBLACK_EX + f"Using options '{options}'" + Fore.RESET)
    return resolved_output_type


def cast_responses_to_datatypes(
    responses: List[Union[str, None]]
) -> List[Union[float, int, str, bool]]:
    responses = [  # type: ignore
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
            responses[idx] = casted_value  # type: ignore
        except (ValueError, SyntaxError, AssertionError):
            continue
    return responses  # type: ignore


def partialclass(cls, *args, **kwds):
    # https://stackoverflow.com/a/38911383
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds)

    NewCls.__name__ = cls.__name__
    return NewCls
