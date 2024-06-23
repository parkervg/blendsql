from typing import Set
import copy
from sqlglot import exp


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


def to_select_star(node) -> exp.Expression:
    """ """
    select_star_node = copy.deepcopy(node)
    select_star_node.find(exp.Select).set(
        "expressions", exp.select("*").args["expressions"]
    )
    return select_star_node
