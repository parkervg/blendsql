import typing as t
from colorama import Fore
from string import Template

from blendsql.ingredients import Ingredient
from blendsql.common.constants import IngredientType
from blendsql.common.logger import logger


def format_ingredient_names_to_lark(
    node_name: str, ingredient_names: t.List[str]
) -> str:
    """Formats list of ingredient names the way our Lark grammar expects.

    Examples:
        ```python
        format_ingredient_names_to_lark(["LLMQA", "LLMVerify"])
        >>> '("LLMQA"i | "LLMVerify"i)'
        ```
    """
    terminal_declarations = "\n".join(
        [f'{name.upper()}: "{name}"i' for name in ingredient_names]
    )
    all_names = "(" + "|".join([n for n in [i.upper() for i in ingredient_names]]) + ")"
    return f"{terminal_declarations}\n{node_name}: {all_names}"


def load_grammar(
    templatized_lark_grammar: str,
    db_schema: t.Optional[dict] = None,
    ingredients: t.Optional[t.Collection[t.Type[Ingredient]]] = None,
) -> str:
    """Loads BlendSQL CFG parser.
    Dynamically modifies grammar string to include only valid ingredients.
    """
    if ingredients is None:
        logger.debug(
            Fore.YELLOW
            + "No ingredients passed to `load_cfg_parser()`!\nWas this on purpose?"
        )
        ingredients = set()
    blendsql_join_functions: t.List[str] = []
    blendsql_aggregate_functions: t.List[str] = []
    blendsql_scalar_functions: t.List[str] = []
    ingredient_type_to_function_type: t.Dict[str, t.List[str]] = {
        IngredientType.JOIN: blendsql_join_functions,
        IngredientType.QA: blendsql_aggregate_functions,
        IngredientType.MAP: blendsql_scalar_functions,
    }

    # Add blendsql function grammar
    for ingredient in ingredients:
        if ingredient.ingredient_type not in ingredient_type_to_function_type:
            # TODO: handle these cases
            continue
        ingredient_type_to_function_type[ingredient.ingredient_type].append(
            ingredient.__name__
        )

    if db_schema is not None:
        # Add database specific grammar
        sorted_tables = sorted(list(db_schema.keys()))
        # wrap_in_valid_quotes = lambda s: f'(DOUBLE_QUOTE "{s}" DOUBLE_QUOTE|"`" "{s}" "`"|"{s}")'
        wrap_in_valid_quotes = lambda s: f'"{s}"'
        quote_list = lambda l: [wrap_in_valid_quotes(item) for item in l]
        STAR = ['"*"']
        column_declarations = [
            f"({' | '.join(quote_list(db_schema[table]) + STAR)})"
            for idx, table in enumerate(sorted_tables)
        ]
        table_vars = [f"table{idx + 1}_refs" for idx in range(len(sorted_tables))]

        column_declaration_grammar_str = "\n".join(
            [
                f'{table_var}: [{wrap_in_valid_quotes(table)} "."] {c}'
                for (table_var, table, c) in zip(
                    table_vars, sorted_tables, column_declarations
                )
            ]
        )
        table_vars.append('"*"')
        column_ref_grammar_str = (
            f"table: ({' | '.join(quote_list(sorted_tables))})"
            + "\n"
            + column_declaration_grammar_str
            + "\n"
            + f"column_ref: ({'|'.join(table_vars)})"
        )
    else:
        column_ref_grammar_str = 'column_ref: [name "."] name\ntable: name'

    lark_grammar: str = Template(templatized_lark_grammar).substitute(
        column_ref_grammar_str=column_ref_grammar_str,
        blendsql_join_functions=format_ingredient_names_to_lark(
            "blendsql_join_functions", blendsql_join_functions
        ),
        blendsql_aggregate_functions=format_ingredient_names_to_lark(
            "blendsql_aggregate_functions", blendsql_aggregate_functions
        ),
        blendsql_scalar_functions=format_ingredient_names_to_lark(
            "blendsql_scalar_functions", blendsql_scalar_functions
        ),
    )
    return lark_grammar
