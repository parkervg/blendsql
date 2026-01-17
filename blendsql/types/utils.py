import itertools
from collections.abc import Collection

from blendsql.common.exceptions import LMFunctionException, TypeResolutionException
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
    return_type: str | DataType | None = None,
    options: Collection[str] | None = None,
    quantifier: QuantifierType | None = None,
    log: bool = True,
) -> DataType:
    # Step 1: Parse string return_type to DataType
    parsed_return_type = _parse_return_type(return_type, quantifier)

    # Step 2: Infer type from options if available
    inferred_from_options = _infer_type_from_options(options, log)

    # Step 3: Resolve final type with precedence rules
    resolved_type = _resolve_final_type(parsed_return_type, inferred_from_options, log)

    # Step 4: Apply quantifier override if provided
    if quantifier:
        resolved_type.quantifier = quantifier

    # Step 5: Handle regex/options conflict
    _handle_regex_options_conflict(resolved_type, options, log)

    return resolved_type


def _parse_return_type(
    return_type: str | DataType | None, quantifier: QuantifierType | None
) -> DataType | None:
    """Convert string return_type to DataType object."""
    if not isinstance(return_type, str):
        return return_type

    return_type = return_type.lower()
    if return_type not in STR_TO_DATATYPE:
        raise LMFunctionException(
            f"{return_type} is not a recognized datatype!\n"
            f"Valid options are {list(STR_TO_DATATYPE.keys())}"
        )

    data_type = STR_TO_DATATYPE[return_type]
    if quantifier:
        data_type.quantifier = quantifier

    return data_type


def _infer_type_from_options(
    options: Collection[str] | None, log: bool
) -> DataType | None:
    """Infer DataType from the provided options collection."""
    if options is None:
        return None

    inferred_type = try_infer_datatype_from_collection(options)
    if inferred_type and log:
        logger.debug(
            Color.quiet_update(
                f"All passed `options` are the same type, so inferring "
                f"a return type of `{inferred_type.name}`"
            )
        )

    return inferred_type


def _resolve_final_type(
    parsed_type: DataType | None, inferred_type: DataType | None, log: bool
) -> DataType:
    """Resolve final type using precedence: options > parsed > default."""
    # If we have an inferred type from options, check if it should override
    if inferred_type is not None:
        if parsed_type is None or parsed_type.atomic_type != inferred_type.atomic_type:
            if parsed_type is not None and parsed_type.name != inferred_type.name:
                raise TypeResolutionException(
                    f"LM function type resolution failed!\nExpression context expects `{parsed_type.name}`, but passed options restrict to `{inferred_type.name}`."
                )
            return inferred_type

    # Fall back to parsed type or default
    if parsed_type is not None:
        return parsed_type

    if inferred_type is not None:
        return inferred_type

    return DataTypes.STR()


def _handle_regex_options_conflict(
    resolved_type: DataType, options: Collection[str] | None, log: bool
) -> None:
    """Handle mutual exclusivity between regex and options."""
    if not log:
        return
    if resolved_type.regex is not None:
        if options is not None:
            logger.debug(
                Color.warning(
                    f"Ignoring inferred regex '{resolved_type.regex}' "
                    f"and using options '{options}' instead!"
                )
            )
            resolved_type.regex = None
        else:
            logger.debug(Color.quiet_update(f"Using regex '{resolved_type.regex}'"))
    elif options:
        logger.debug(
            Color.quiet_update(
                f"Using options '{set(itertools.islice(options, 20))}...'"
            )
        )


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
            if return_type.atomic_type == "str":
                return [
                    return_type.coerce_fn(c, db)
                    for c in ast.literal_eval(re.sub(r"(\w)'(\w)", r"\1\\'\2", s))
                ]

    else:
        return return_type.coerce_fn(s, db)
