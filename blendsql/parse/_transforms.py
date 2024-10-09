"""
Defines a set of transformations on the SQL AST, to be used with sqlglot.
https://github.com/tobymao/sqlglot

"""

from typing import Union
from sqlglot import exp
from sqlglot.optimizer.scope import find_in_scope

from . import _checks as check
from ._constants import SUBQUERY_EXP
from ._dialect import _parse_one, FTS5SQLite


def extract_multi_table_predicates(
    node: exp.Where, tablename: str
) -> Union[exp.Expression, None]:
    """Extracts all non-Column predicates acting on a given tablename.
    non-Column since we want to exclude JOIN's (e.g. JOIN on A.col = B.col)

    Requirements to keep:
        - Must be a predicate node with an expression arg containing column associated with tablename, and some other non-column arg
        - Or, we're in a subquery
    If we have a predicate not meeting these conditions, set to exp.true()
        - This is much simpler than doing surgery on `WHERE AND ...` sort of relics
        - TODO: is the above true? Maybe simpler to actually fully remove vs. set to true?

    Used with node.transform()

    Args:
        node: The exp.Where clause we're extracting predicates from
        tablename: The name of the table whose predicates we keep
    """
    if isinstance(node, exp.Where):
        return node
    # Don't abstract to `TRUE` if we're in a subquery
    # This is important!!! Without this, test_multi_table_blendsql.test_simple_multi_exec will fail
    # Causes difference between `SELECT * FROM portfolio WHERE TRUE AND portfolio.Symbol IN (SELECT Symbol FROM constituents WHERE constituents.Sector = 'Information Technology')`
    #   and `SELECT * FROM portfolio WHERE TRUE AND portfolio.Symbol IN (SELECT Symbol FROM constituents WHERE TRUE)`
    if check.in_subquery(node):
        return node
    if isinstance(node, exp.Table):
        if node.name != tablename:
            return None
    if isinstance(node, (exp.Predicate, exp.Unary)):
        # Search for child table, in current view
        # This below ensures we don't go into subqueries
        child_table = find_in_scope(node, exp.Table)
        if child_table is not None and child_table.name != tablename:
            return None
        if "this" in node.args:
            # Need to apply `find` here in case of `LOWER` arg getting in our way
            this_column = (
                node.args["this"].find(exp.Column) if "this" in node.args else None
            )
            expression_column = (
                node.args["expression"].find(exp.Column)
                if "expression" in node.args
                else None
            )
            if this_column is None:
                return node
            if this_column.table == tablename:
                # This is true if we have a subquery as a 2nd arg
                # Just leave this as-is
                if "query" in node.args:
                    return node
                if expression_column is None:
                    return node
                # This is False if we have a `JOIN` (a.colname = b.colname)
                elif expression_column.table == "":
                    return node
            else:
                # If the expression is a self-contained subquery
                # 'self-contained' means it starts with its own `SELECT`
                expression_args = (
                    node.args["expression"] if "expression" in node.args else None
                )
                if expression_args is None:
                    return node
                if isinstance(expression_args, exp.Subquery) and expression_args.find(
                    exp.Select
                ):
                    return node
        return exp.true()
    return node


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


def prune_with(node):
    """
    Removes any exp.With nodes.

    Used with node.transform()
    """
    if isinstance(node, exp.With):
        return None
    return node


def prune_true_where(node):
    """
    Removes artifacts like `WHERE TRUE AND TRUE`

    Used with node.transform()
    """
    if isinstance(node, exp.Where):
        if isinstance(node.args["this"], exp.Connector):
            values_to_check = set(node.args["this"].args.values())
        else:
            values_to_check = set([node.args["this"]])
        if values_to_check == {exp.true()}:
            return None
    return node


def set_structs_to_true(node) -> Union[exp.Expression, None]:
    """Prunes all nodes with an exp.Struct parent.

    CASE 1
    Turns the exp.Struct node itself to a TRUE.

    CASE 2
    In the case below ('x = {{A()}}'):
        exp.Condition
             / \
           x    exp.Struct
    We need to set the whole exp.Condition clause to TRUE.

    Used with node.transform()
    """
    # Case 1: we have an exp.Struct in isolation
    if isinstance(node, exp.Struct):
        return exp.true()
    # Case 2: we have an exp.Struct within a predicate (=, <, >, etc.)
    if isinstance(node, exp.Predicate):
        if any(
            isinstance(x, exp.Struct)
            for x in {node.args.get("this", None), node.args.get("expression", None)}
        ):
            return exp.true()
    return node


def replace_join_with_ingredient_multiple_ingredient(
    node: exp.Where, ingredient_name: str, ingredient_alias: str, temp_uuid: str
) -> Union[exp.Expression, None]:
    """

    Used with node.transform()

    sqlglot re-orders `WHERE` conditions to appear in `JOIN`:

    SELECT * FROM documents JOIN "w" ON {{B()}} WHERE w.film = {{A()}}
    SELECT * FROM documents JOIN "w" ON w.film = {{A()}}  AND  {{B()}}  WHERE TRUE
    """
    if isinstance(node, exp.Join):
        anon_child_nodes = node.find_all(exp.Anonymous)
        to_return = []
        join_alias: str = ""
        for anon_child_node in anon_child_nodes:
            if anon_child_node.name == ingredient_name:
                join_alias = ingredient_alias
                continue
            # Traverse and get the whole ingredient
            # We need to go up 2 parents
            _parent = anon_child_node
            for _ in range(2):
                _parent = _parent.parent
                assert isinstance(_parent, exp.Expression)
            to_return.append(_parent.sql(dialect=FTS5SQLite))
        if len(to_return) == 0:
            return node
        if join_alias == "":
            raise ValueError
        # temp_uuid is used to ensure a partial query that is parse-able by sqlglot
        # This gets removed after
        return _parse_one(
            f' SELECT "{temp_uuid}", '
            + join_alias
            + " WHERE "
            + " AND ".join(to_return)
        )
    return node


def replace_join_with_ingredient_single_ingredient(
    node: exp.Where, ingredient_name: str, ingredient_alias: str
) -> Union[exp.Expression, None]:
    """

    Used with node.transform()
    """
    if isinstance(node, exp.Join):
        anon_child_node = node.find(exp.Anonymous)
        if anon_child_node is not None:
            if anon_child_node.name == ingredient_name:
                return _parse_one(f" {ingredient_alias}")
    return node


def replace_subquery_with_direct_alias_call(node, subquery, aliasname):
    """

    Used with node.transform()
    """
    if node == subquery:
        return exp.Table(this=exp.Identifier(this=aliasname))
    return node


def remove_ctes(node):
    if isinstance(node, exp.With):
        return None
    return node


def maybe_set_subqueries_to_true(node):
    if len([i for i in node.find_all(SUBQUERY_EXP + (exp.Paren,))]) == 1:
        return node
    return node.transform(set_subqueries_to_true).transform(prune_empty_where)
