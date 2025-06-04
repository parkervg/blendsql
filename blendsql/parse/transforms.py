"""
Defines a set of transformations on the SQL AST, to be used with sqlglot.
https://github.com/tobymao/sqlglot

"""

from typing import Union, Type, Tuple

from sqlglot import exp

from . import checks as check
from .constants import SUBQUERY_EXP


def set_subqueries_to_true(node) -> Union[exp.Expression, None]:
    """For all subqueries (i.e. children exp.Select statements)
    set all these to TRUE abstractions.

    Used with node.transform().
    """
    if isinstance(node, exp.Predicate):
        if all(x in node.args for x in {"this", "expression"}):
            if node.args["expression"].find(exp.Subquery):
                if node.args["this"].find(exp.Select) is None:
                    return node.args["this"]
                else:
                    return None
        if check.in_subquery(node):
            return None
    if isinstance(node, SUBQUERY_EXP + (exp.Paren,)) and node.parent is not None:
        return exp.true()
    parent_select = node.find_ancestor(SUBQUERY_EXP)
    if parent_select and parent_select.parent is not None:
        return None
    return node


def prune_empty_where(node) -> Union[exp.Expression, None]:
    """
    Removes any `exp.Where` clause without any values.

    Used with node.transform()
    """
    if isinstance(node, exp.Where):
        if set(node.args.values()) == {None}:
            return None
        elif "this" in node.args:
            where_arg = node.args["this"]
            if "query" in where_arg.args and isinstance(
                where_arg.args["query"], exp.Boolean
            ):
                return None
    # Don't check *all* predicates here
    # Since 'WHERE a = TRUE' is valid and should be kept
    elif isinstance(node, exp.In):
        if "query" in node.args:
            if isinstance(node.args["query"], exp.Boolean):
                return None
    return node


def set_ingredient_nodes_to_true(node) -> Union[exp.Expression, None]:
    """Prunes all nodes with an ingredient parent.

    CASE 1
    Turns the ingredient node itself to a TRUE (`SELECT * WHERE {{A()}}`).

    CASE 2
    In the case below ('x = {{A()}}'):
        exp.Condition
             / \
           x    ingredient
    We need to set the whole exp.Condition clause to TRUE.

    Used with node.transform()
    """
    # Case 1: we have an Ingredient in isolation
    if isinstance(node, exp.BlendSQLFunction):
        return exp.true()
    # Case 2: we have an Ingredient within a predicate (=, <, >, IN, etc.)
    if isinstance(node, exp.Predicate):
        # Traverse over all nodes in predicate args,
        #   to handle case when we have nested function calls
        #   Example: `LENGTH(UPPER({{LLMMap()}})) > 3`
        if node.find(exp.BlendSQLFunction):
            return exp.true()
    return node


def replace_subquery_with_direct_alias_call(node, subquery, aliasname):
    """

    Used with node.transform()
    """
    if node == subquery:
        return exp.Table(this=exp.Identifier(this=aliasname))
    return node


def remove_nodetype(
    node, nodetype: Union[Type[exp.Expression], Tuple[Type[exp.Expression]]]
):
    if isinstance(node, nodetype):
        return None
    return node


def maybe_set_subqueries_to_true(node):
    if len([i for i in node.find_all(SUBQUERY_EXP + (exp.Paren,))]) == 1:
        return node
    return node.transform(set_subqueries_to_true).transform(prune_empty_where)


def replace_tablename(node, original_tablename, new_tablename):
    if isinstance(node, exp.Table):
        if node.name.lower() == original_tablename.lower():
            node.set("this", exp.Identifier(this=new_tablename, quoted=True))
        if "alias" in node.args and node.args["alias"].name == original_tablename:
            node.args["alias"].set(
                "this", exp.Identifier(this=new_tablename, quoted=True)
            )
    elif isinstance(node, exp.Column) and "table" in node.args:
        if node.args["table"].name.lower() == original_tablename.lower():
            node.set("table", exp.Identifier(this=new_tablename, quoted=True))
    return node
