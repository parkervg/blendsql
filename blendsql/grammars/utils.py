from pathlib import Path
from typing import Type
from collections.abc import Collection
from string import Template

from blendsql.common.logger import logger, Color
from ..ingredients import Ingredient
from blendsql.common.typing import IngredientType
from .minEarley.parser import EarleyParser


def format_ingredient_names_to_lark(names: list[str]) -> str:
    """Formats list of ingredient names the way our Lark grammar expects.

    Examples:
        ```python
        format_ingredient_names_to_lark(["LLMQA", "LLMVerify"])
        >>> '("LLMQA("i | "LLMVerify("i)'
        ```
    """
    return "(" + " | ".join([f'"{n}("i' for n in names]) + ")"


def load_cfg_parser(
    ingredients: Collection[Type[Ingredient]] | None = None,
) -> EarleyParser:
    """Loads BlendSQL CFG parser.
    Dynamically modifies grammar string to include only valid ingredients.
    """
    if ingredients is None:
        logger.debug(
            Color.warning(
                "No ingredients passed to `load_cfg_parser()`!\nWas this on purpose?"
            )
        )
        ingredients = set()
    with open(Path(__file__).parent / "./cfg_grammar.lark", encoding="utf-8") as f:
        cfg_grammar = Template(f.read())
    blendsql_join_functions: list[str] = []
    blendsql_aggregate_functions: list[str] = []
    blendsql_scalar_functions: list[str] = []
    ingredient_type_to_function_type: dict[str, list[str]] = {
        IngredientType.JOIN: blendsql_join_functions,
        IngredientType.QA: blendsql_aggregate_functions,
        IngredientType.MAP: blendsql_scalar_functions,
    }
    for ingredient in ingredients:
        if ingredient.ingredient_type not in ingredient_type_to_function_type:
            # TODO: handle these cases
            continue
        ingredient_type_to_function_type[ingredient.ingredient_type].append(
            ingredient.__name__
        )
    cfg_grammar_str: str = cfg_grammar.substitute(
        blendsql_join_functions=format_ingredient_names_to_lark(
            blendsql_join_functions
        ),
        blendsql_aggregate_functions=format_ingredient_names_to_lark(
            blendsql_aggregate_functions
        ),
        blendsql_scalar_functions=format_ingredient_names_to_lark(
            blendsql_scalar_functions
        ),
    )
    return EarleyParser(
        grammar=cfg_grammar_str,
        start="start",
        keep_all_tokens=True,
    )
