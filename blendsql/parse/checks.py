# This file contains sqlglot-based functions that return a boolean
from sqlglot import exp
from functools import lru_cache

from .constants import SUBQUERY_EXP
from .utils import get_first_child


def get_ingredient_count(node) -> int:
    return len(get_ingredient_nodes(node))


def get_ingredient_nodes(node) -> list:
    return list(node.find_all(exp.BlendSQLFunction))


def all_terminals_are_true(node: exp.Expression) -> bool:
    """
    Check if all terminal nodes of a given node are TRUE booleans.

    Args:
        node (exp.Expression): The root expression node to check.

    Returns:
        bool: True if all terminal nodes are TRUE booleans, False otherwise.
    """
    for n in node.walk():
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
        if p.find(exp.CTE):
            table_alias = p.find(exp.TableAlias)
            if table_alias is not None:
                return (True, table_alias.name) if return_name else True
    return (False, None) if return_name else False


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
        if n.find(exp.BlendSQLFunction):
            if n.alias in ref_columns:
                return True

    return False
