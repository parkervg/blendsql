import copy
from sqlglot import exp

from blendsql.db.utils import double_quote_escape


def get_first_child(node):
    """
    Helper function to get first child of a node.
    The default argument to `walk()` is bfs=True,
    meaning we do breadth-first search.
    """
    gen = node.walk()
    _ = next(gen)
    return next(gen)


def set_select_to(
    node,
    tablenames: list[str],
    columnnames: list[str],
    aliasnames: list[str] | None = None,
) -> exp.Expression:
    assert len(tablenames) == len(columnnames)
    select_star_node = copy.deepcopy(node)
    to_select = [
        f'"{double_quote_escape(tablename)}"."{double_quote_escape(columnname)}"'
        for tablename, columnname in zip(tablenames, columnnames)
    ]
    if aliasnames is not None:
        to_select = [
            ts + f' AS "{double_quote_escape(aliasname)}"'
            for ts, aliasname in zip(to_select, aliasnames)
        ]
    select_star_node.find(exp.Select).set(
        "expressions", exp.select(*to_select).args["expressions"]
    )
    return select_star_node
