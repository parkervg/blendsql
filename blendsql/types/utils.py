import itertools
from collections.abc import Collection

from blendsql.common.exceptions import IngredientException
from blendsql.types.types import (
    DataTypes,
    STR_TO_DATATYPE,
)
from blendsql.db import Database
from blendsql.common.typing import QuantifierType, DataType
from blendsql.common.logger import logger, Color


def prepare_datatype(
    options: Collection[str] | None,
    return_type: str | DataType | None = None,
    quantifier: QuantifierType | None = None,
) -> DataType:
    if return_type is None:
        resolved_output_type = DataTypes.ANY()
    elif isinstance(return_type, str):
        # The user has passed us an output type in the BlendSQL query
        # That should take precedence
        return_type = return_type.lower()
        if return_type not in STR_TO_DATATYPE:
            raise IngredientException(
                f"{return_type} is not a recognized datatype!\nValid options are {list(STR_TO_DATATYPE.keys())}"
            )
        resolved_output_type = STR_TO_DATATYPE[return_type]
        if quantifier:  # User passed quantifier takes precedence
            resolved_output_type.quantifier = quantifier
    elif isinstance(return_type, DataType):
        resolved_output_type = return_type
    if quantifier:
        # The user has passed us a quantifier that should take precedence
        resolved_output_type.quantifier = quantifier
    if resolved_output_type.regex is not None:
        if options is not None:
            logger.debug(
                Color.quiet_update(
                    f"Ignoring inferred regex '{resolved_output_type.regex}' and using options '{options}' instead"
                )
            )
            resolved_output_type.regex = None
        else:
            logger.debug(
                Color.quiet_update(f"Using regex '{resolved_output_type.regex}'")
            )
    elif options:
        logger.debug(
            Color.quiet_update(
                f"Using options '{set(itertools.islice(options, 20))}...'"
            )
        )
    return resolved_output_type


def apply_type_conversion(s: str, return_type: DataType, db: Database):
    import ast
    import re

    is_list_output = return_type.quantifier is not None
    if is_list_output:
        try:
            return [return_type.coerce_fn(c, db) for c in ast.literal_eval(s)]
        except Exception:
            # Sometimes we need to first escape single quotes
            # E.g. in ['Something's wrong here']
            if return_type.name == "str":
                return [
                    [
                        return_type.coerce_fn(c, db)
                        for c in ast.literal_eval(re.sub(r"(\w)'(\w)", r"\1\\'\2", s))
                    ]
                ]

    else:
        return return_type.coerce_fn(s, db)
