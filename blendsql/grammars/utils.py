from pathlib import Path
from typing import Optional, List, Dict, Type
from collections.abc import Collection
from string import Template
from colorama import Fore

from .._logger import logger
from ..ingredients import Ingredient
from .._constants import IngredientType
from .minEarley.parser import EarleyParser


def format_ingredient_names_to_lark(names: List[str]) -> str:
    """Formats list of ingredient names the way our Lark grammar expects.

    Examples:
        ```python
        format_ingredient_names_to_lark(["LLMQA", "LLMVerify"])
        >>> '("LLMQA("i | "LLMVerify("i)'
        ```
    """
    return "(" + " | ".join([f'"{n}("i' for n in names]) + ")"


def load_cfg_parser(
    ingredients: Optional[Collection[Type[Ingredient]]] = None,
) -> EarleyParser:
    """Loads BlendSQL CFG parser.
    Dynamically modifies grammar string to include only valid ingredients.
    """
    if ingredients is None:
        logger.debug(
            Fore.YELLOW
            + "No ingredients passed to `load_cfg_parser()`!\nWas this on purpose?"
        )
        ingredients = set()
    with open(Path(__file__).parent / "./_cfg_grammar.lark", encoding="utf-8") as f:
        cfg_grammar = Template(f.read())
    blendsql_join_functions: List[str] = []
    blendsql_aggregate_functions: List[str] = []
    blendsql_scalar_functions: List[str] = []
    ingredient_type_to_function_type: Dict[str, List[str]] = {
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


if __name__ == "__main__":
    from blendsql import LLMMap, LLMJoin, LLMValidate

    parser = load_cfg_parser({LLMMap, LLMJoin, LLMValidate})
    parser.parse("{{LLMQA('what is the answer', (select * from w))}}")
    print()
