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


def try_infer_datatype_from_collection(collection: Collection) -> DataType:
    if all(isinstance(x, str) for x in collection):
        return DataTypes.STR()
    elif all(isinstance(x, bool) for x in collection):
        return DataTypes.BOOL()
    elif all(isinstance(x, float) for x in collection):
        return DataTypes.FLOAT()
    elif all(isinstance(x, int) for x in collection):
        return DataTypes.INT()
    return None


def prepare_datatype(
    options: Collection[str] | None,
    return_type: str | DataType | None = None,
    quantifier: QuantifierType | None = None,
    log: bool = True,
) -> DataType:
    resolved_return_type = None
    if options is not None:
        # We can still infer a return_type from the options we got
        # This should take precedence over the expression-inferred return type
        resolved_return_type: DataType | None = try_infer_datatype_from_collection(
            options
        )
        if resolved_return_type is not None:
            if log:
                logger.debug(
                    Color.quiet_update(
                        f"All passed `options` are the same type, so inferring a return type of `{resolved_return_type.name}`'"
                    )
                )
            if return_type is not None:
                if return_type.name != resolved_return_type.name:
                    if log:
                        logger.debug(
                            Color.quiet_update(
                                f"This will override the expression-inferred return type of `{return_type.name}`'"
                            )
                        )
                return_type = None

    if return_type is None and resolved_return_type is None:
        # Use default base of `DataTypes.STR`
        resolved_return_type = DataTypes.STR()

    elif isinstance(return_type, str):
        # The user has passed us an output type in the BlendSQL query
        # That should take precedence
        return_type = return_type.lower()
        if return_type not in STR_TO_DATATYPE:
            raise IngredientException(
                f"{return_type} is not a recognized datatype!\nValid options are {list(STR_TO_DATATYPE.keys())}"
            )
        resolved_return_type = STR_TO_DATATYPE[return_type]
        if quantifier:  # User passed quantifier takes precedence
            resolved_return_type.quantifier = quantifier
    elif isinstance(return_type, DataType):
        resolved_return_type = return_type

    if quantifier:
        # The user has passed us a quantifier that should take precedence
        resolved_return_type.quantifier = quantifier

    if resolved_return_type.regex is not None:
        if options is not None:
            if log:
                logger.debug(
                    Color.quiet_update(
                        f"Ignoring inferred regex '{resolved_return_type.regex}' and using options '{options}' instead"
                    )
                )
            resolved_return_type.regex = None
        else:
            if log:
                logger.debug(
                    Color.quiet_update(f"Using regex '{resolved_return_type.regex}'")
                )
    elif options:
        if log:
            logger.debug(
                Color.quiet_update(
                    f"Using options '{set(itertools.islice(options, 20))}...'"
                )
            )
    return resolved_return_type


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
