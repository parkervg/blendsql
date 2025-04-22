from typing import Set
import copy
from sqlglot import exp

from ..db.utils import double_quote_escape


def get_first_child(node):
    """
    Helper function to get first child of a node.
    The default argument to `walk()` is bfs=True,
    meaning we do breadth-first search.
    """
    gen = node.walk()
    _ = next(gen)
    return next(gen)[0]


def get_alias_identifiers(node) -> Set[str]:
    """Given a SQL statement, returns defined aliases.
    Examples:
        >>> get_alias_identifiers(_parse_one("SELECT {{LLMMap('year from date', 'w::date')}} AS year FROM w")
        ['year']
    """
    return set([i.find(exp.Identifier).name for i in node.find_all(exp.Alias)])


def set_select_to(node, tablename, columnnames) -> exp.Expression:
    select_star_node = copy.deepcopy(node)
    to_select = [
        f'"{double_quote_escape(tablename)}"."{double_quote_escape(columnname)}"'
        for columnname in columnnames
    ]
    select_star_node.find(exp.Select).set(
        "expressions", exp.select(*to_select).args["expressions"]
    )
    return select_star_node
