# This file contains sqlglot-based functions that return a boolean
from sqlglot import exp

from ._constants import SUBQUERY_EXP
from ._utils import get_first_child


def all_terminals_are_true(node) -> bool:
    """Check to see if all terminal nodes of a given node are TRUE booleans."""
    for n, _, _ in node.walk():
        try:
            get_first_child(n)
        except StopIteration:
            if n != exp.true():
                return False
    return True


def ingredients_only_in_top_select(node) -> bool:
    select_exps = list(node.find_all(exp.Select))
    if len(select_exps) == 1:
        # Check if the only `STRUCT` nodes are found in select
        all_struct_exps = list(node.find_all(exp.Struct))
        if len(all_struct_exps) > 0:
            num_select_struct_exps = sum(
                [
                    len(list(n.find_all(exp.Struct)))
                    for n in select_exps[0].find_all(exp.Alias)
                ]
            )
            if num_select_struct_exps == len(all_struct_exps):
                return True
    return False


def in_subquery(node) -> bool:
    _ancestor = node.find_ancestor(SUBQUERY_EXP)
    if _ancestor is not None:
        return _ancestor.find_ancestor(SUBQUERY_EXP) is not None
    return False


def in_cte(node, return_name: bool = False):
    p = node.parent
    if p is not None:
        table_alias = p.find(exp.TableAlias)
        if table_alias is not None:
            return (True, table_alias.name) if return_name else True
    return (False, None) if return_name else False
