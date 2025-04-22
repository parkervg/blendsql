import itertools
import typing as t
from colorama import Fore
from collections.abc import Collection

from blendsql.common.exceptions import IngredientException
from blendsql.types.types import (
    ModifierType,
    DataType,
    DataTypes,
    STR_TO_DATATYPE,
)
from blendsql.common.logger import logger


def prepare_datatype(
    options: t.Optional[Collection[str]],
    output_type: t.Optional[t.Union[str, DataType]] = None,
    modifier: t.Optional[ModifierType] = None,
) -> DataType:
    if output_type is None:
        resolved_output_type = DataTypes.STR()
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
        logger.debug(
            Fore.LIGHTBLACK_EX
            + f"Using options '{set(itertools.islice(options, 20))}...'"
            + Fore.RESET
        )
    return resolved_output_type
