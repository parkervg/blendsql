# This file contains sqlglot-based functions that return a boolean
from sqlglot import exp
from functools import lru_cache
import re

from ._constants import SUBQUERY_EXP
from ._utils import get_first_child

INGREDIENT_PATTERN = re.compile("{{[A-Z]\(\)}}")


def get_ingredient_count(node) -> int:
    return len(
        list(filter(lambda x: is_ingredient_node(x), node.find_all(exp.Identifier)))
    )


def is_blendsql_query(s: str) -> bool:
    """A quick function to determine if the given string is a query that needs
    to be executed.
    """
    return s.upper().startswith(("SELECT", "WITH", "{{"))


def is_ingredient_node(node: exp.Expression) -> bool:
    """Checks to see if a given node is pointing to an ingredient.

    We need to handle both exp.Identifier and exp.Column types below,
    since sqlglot will interpret the exp.Function type different depending
    on context.

    For example:
    > SELECT {{B()}} FROM table WHERE a IN {{A()}}

    {{B()}} will get parsed as (COLUMN this:
    (IDENTIFIER this: {{B()}}, quoted: False))

    But {{A()}} will get parsed as (IDENTIFIER this: {{A()}}, quoted: False)
    """
    if not isinstance(node, (exp.Identifier, exp.Column)):
        return False
    if isinstance(node, exp.Column):
        node = node.find(exp.Identifier)
    return INGREDIENT_PATTERN.match(node.this) is not None


def ingredient_node_in_ancestors(node: exp.Expression) -> bool:
    ancestor = node.find_ancestor(exp.Identifier)
    if ancestor and INGREDIENT_PATTERN.match(ancestor.this):
        return True
    return False


def all_terminals_are_true(node: exp.Expression) -> bool:
    """
    Check if all terminal nodes of a given node are TRUE booleans.

    Args:
        node (exp.Expression): The root expression node to check.

    Returns:
        bool: True if all terminal nodes are TRUE booleans, False otherwise.
    """
    for n, _, _ in node.walk():
        try:
            get_first_child(n)
        except StopIteration:
            if n != exp.true():
                return False
    return True


def ingredients_only_in_top_select(node: exp.Expression) -> bool:
    """
    Check if all `STRUCT` nodes are only found in the top-level SELECT statement.

    Args:
        node (exp.Expression): The root expression node to check.

    Returns:
        bool: True if all `STRUCT` nodes are only found in the top-level SELECT statement, False otherwise.
    """
    select_exps = list(node.find_all(exp.Select))
    if len(select_exps) == 1:
        # Check if the only ingredient nodes are found in top-level select expression
        num_ingredients_in_select = sum(
            [get_ingredient_count(i) for i in select_exps[0].expressions]
        )
        if num_ingredients_in_select == 0:
            return False
        all_ingredient_count = get_ingredient_count(node)
        if num_ingredients_in_select == all_ingredient_count:
            return True
    return False


def in_subquery(node: exp.Expression) -> bool:
    _ancestor = node.find_ancestor(SUBQUERY_EXP)
    if _ancestor is not None:
        return _ancestor.find_ancestor(SUBQUERY_EXP) is not None
    return False


def in_cte(node: exp.Expression, return_name: bool = False):
    p = node.parent
    if p is not None:
        table_alias = p.find(exp.TableAlias)
        if table_alias is not None:
            return (True, table_alias.name) if return_name else True
    return (False, None) if return_name else False


def contains_ingredient(node: exp.Expression) -> bool:
    for n in node.find_all(exp.Identifier):
        if is_ingredient_node(n):
            return True
    return False


def ingredient_alias_in_query_body(node: exp.Expression) -> bool:
    """Check if an alias created from an ingredient is used in the main query body.

    Examples:
        ```sql
        SELECT Name,
        {{LLMMap('Contains a superlative?', 'parks::Description')}} AS "Contains Superlative"
        FROM parks
        GROUP BY "Contains Superlative"
        ```
        Returns `True`

        ```sql
        SELECT Name,
        {{LLMMap('Contains a superlative?', 'parks::Description')}} AS "Contains Superlative"
        FROM parks
        ```
        Returns `False`
    """
    # TODO: is there a way to return false if generator is empty

    @lru_cache
    def get_referenced_columns(node: exp.Expression) -> set:
        """Returns set of all referenced columns in body of query
        (i.e. not in select statement)
        """
        return set([i.name for i in node.find_all(exp.Column)])

    for n in node.find_all(exp.Alias):
        ref_columns = get_referenced_columns(node)
        if contains_ingredient(n):
            if n.alias in ref_columns:
                return True

    return False
