from pathlib import Path
from typing import Optional, Collection, List, Dict
from string import Template

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


def load_cfg_parser(ingredients: Optional[Collection[Ingredient]]) -> EarleyParser:
    """Loads BlendSQL CFG parser.
    Dynamically modifies grammar string to include only valid ingredients.
    """
    with open(Path(__file__).parent / "./_cfg_grammar.lark", encoding="utf-8") as f:
        cfg_grammar = Template(f.read())
    blendsql_join_functions = []
    blendsql_aggregate_functions = []
    ingredient_type_to_function_type: Dict[str, List[str]] = {
        IngredientType.JOIN: blendsql_join_functions,
        IngredientType.QA: blendsql_aggregate_functions,
    }
    for ingredient in ingredients:
        if ingredient.ingredient_type not in ingredient_type_to_function_type:
            print(
                f"Not sure what to do with ingredient type '{ingredient.ingredient_type}'"
            )
            continue
        ingredient_type_to_function_type[ingredient.ingredient_type].append(
            ingredient.__name__
        )
    cfg_grammar = cfg_grammar.substitute(
        blendsql_join_functions=format_ingredient_names_to_lark(
            blendsql_join_functions
        ),
        blendsql_aggregate_functions=format_ingredient_names_to_lark(
            blendsql_aggregate_functions
        ),
    )
    return EarleyParser(
        grammar=cfg_grammar,
        start="start",
        keep_all_tokens=True,
    )


if __name__ == "__main__":
    from blendsql import LLMMap, LLMJoin, LLMValidate

    parser = load_cfg_parser({LLMMap, LLMJoin, LLMValidate})
    # parser.parse("{{LLMQA('what is the answer', (select * from w))}}")
    print()
