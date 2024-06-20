import sqlglot
from sqlglot import exp, Schema
from sqlglot.optimizer.scope import build_scope
from typing import (
    Generator,
    List,
    Set,
    Tuple,
    Union,
    Callable,
    Type,
    Optional,
    Dict,
    Any,
    Literal,
)
from ast import literal_eval
from sqlglot.optimizer.scope import find_all_in_scope, find_in_scope
from attr import attrs, attrib

from .utils import recover_blendsql
from ._constants import DEFAULT_ANS_SEP, DEFAULT_NAN_ANS
from ._dialect import _parse_one, FTS5SQLite
from ._logger import logger

"""
Defines a set of transformations on the SQL AST, to be used with sqlglot.
https://github.com/tobymao/sqlglot

sqlglot.optimizer.simplify looks interesting
"""

SUBQUERY_EXP = (exp.Select,)
CONDITIONS = (
    exp.Where,
    exp.Group,
    # IMPORTANT: If we uncomment limit, then `test_limit` in `test_single_table_blendsql.py` will not pass
    # exp.Limit,
    exp.Except,
    exp.Order,
)
MODIFIERS = (
    exp.Delete,
    exp.AlterColumn,
    exp.AlterTable,
    exp.Drop,
    exp.RenameTable,
    exp.Drop,
    exp.DropPartition,
)


def is_in_subquery(node):
    _ancestor = node.find_ancestor(SUBQUERY_EXP)
    if _ancestor is not None:
        return _ancestor.find_ancestor(SUBQUERY_EXP) is not None
    return False


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
        if is_in_subquery(node):
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
            # We need to go up 3 parents
            _parent = anon_child_node
            for _ in range(3):
                _parent = _parent.parent
                assert isinstance(_parent, exp.Expression)
            to_return.append(_parent.sql(dialect=FTS5SQLite))
        if len(to_return) == 0:
            return node
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
    if is_in_subquery(node):
        return node
    if isinstance(node, exp.Table):
        if node.name != tablename:
            return None
    if isinstance(node, exp.Predicate):
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


def get_predicate_literals(node) -> List[str]:
    """From a given SQL clause, gets all literals appearing as object of predicate.
    (We treat booleans as literals here, which might be a misuse of terminology.)

    Examples:
        >>> get_predicate_literals(_parse_one("{{Model('year', 'w::year')}} IN ('2010', '2011', '2012')"))
        ['2010', '2011', '2012']
    """
    literals = set()
    gen = node.walk()
    _ = next(gen)
    if isinstance(node.parent, exp.Select):
        return []
    for child, _, _ in gen:
        if child.find_ancestor(exp.Struct) or isinstance(child, exp.Struct):
            continue
        if isinstance(child, exp.Literal):
            literals.add(
                literal_eval(child.name) if not child.is_string else child.name
            )
            continue
        elif isinstance(child, exp.Boolean):
            literals.add(child.args["this"])
            continue
        for i in child.find_all(exp.Literal):
            if i.find_ancestor(exp.Struct):
                continue
            literals.add(literal_eval(i.name) if not i.is_string else i.name)
    return list(literals)


def get_reversed_subqueries(node):
    """Iterates through all subqueries (either parentheses or select).
    Reverses all EXCEPT for CTEs, which should remain in order.
    """
    # First, fetch all common table expressions
    r = [i for i in node.find_all(SUBQUERY_EXP + (exp.Paren,)) if is_in_cte(i)]
    # Then, add (reversed) other subqueries
    return (
        r
        + [i for i in node.find_all(SUBQUERY_EXP + (exp.Paren,)) if not is_in_cte(i)][
            ::-1
        ]
    )


def replace_subquery_with_direct_alias_call(node, subquery, aliasname):
    """

    Used with node.transform()
    """
    if node == subquery:
        return exp.Table(this=exp.Identifier(this=aliasname))
    return node


def is_in_cte(node, return_name: bool = False):
    p = node.parent
    if p is not None:
        table_alias = p.find(exp.TableAlias)
        if table_alias is not None:
            return (True, table_alias.name) if return_name else True
    return (False, None) if return_name else False


def remove_ctes(node):
    if isinstance(node, exp.With):
        return None
    return node


def maybe_set_subqueries_to_true(node):
    if len([i for i in node.find_all(SUBQUERY_EXP + (exp.Paren,))]) == 1:
        return node
    return node.transform(set_subqueries_to_true).transform(prune_empty_where)


def all_terminals_are_true(node) -> bool:
    """Check to see if all terminal nodes of a given node are TRUE booleans."""
    for n, _, _ in node.walk():
        try:
            get_first_child(n)
        except StopIteration:
            if n != exp.true():
                return False
    return True


def get_scope_nodes(
    nodetype: Type[exp.Expression],
    restrict_scope: bool = False,
    root: Optional[sqlglot.optimizer.Scope] = None,
    node: Optional[exp.Expression] = None,
) -> Generator:
    """Utility to get nodes of a certain type within our subquery scope.

    https://github.com/tobymao/sqlglot/blob/v20.9.0/posts/ast_primer.md#scope
    """
    if root is None:
        assert node is not None
        root = build_scope(node)

    if restrict_scope:
        for tablenode in find_all_in_scope(root.expression, nodetype):
            yield tablenode
    else:
        for tablenode in [
            source
            for scope in root.traverse()
            for alias, (node, source) in scope.selected_sources.items()
            if isinstance(source, nodetype)
        ]:
            yield tablenode


@attrs
class QueryContextManager:
    """Handles manipulation of underlying SQL query.
    We need to maintain two synced representations here:

        1) The underlying sqlglot exp.Expression node

        2) The string representation of the query
    """

    node: exp.Expression = attrib(default=None)
    _query: str = attrib(default=None)
    _last_to_string_node: exp.Expression = None

    def parse(self, query, schema: Optional[Union[dict, Schema]] = None):
        self._query = query
        self.node = _parse_one(query, schema=schema)

    def to_string(self):
        # Only call `recover_blendsql` if we need to
        if hash(self.node) != hash(self._last_to_string_node):
            self._query = recover_blendsql(self.node.sql(dialect=FTS5SQLite))
            self.last_to_string_node = self.node
        return self._query

    def __setattr__(self, name, value):
        self.__dict__[name] = value


@attrs
class SubqueryContextManager:
    node: exp.Select = attrib()
    prev_subquery_has_ingredient: bool = attrib()
    tables_in_ingredients: set = attrib()

    # Keep a running log of what aliases we've initialized so far, per subquery
    alias_to_subquery: dict = attrib(default=None)
    alias_to_tablename: dict = attrib(init=False)
    tablename_to_alias: dict = attrib(init=False)
    root: sqlglot.optimizer.scope.Scope = attrib(init=False)

    def __attrs_post_init__(self):
        self.alias_to_tablename = {}
        self.tablename_to_alias = {}
        # https://github.com/tobymao/sqlglot/blob/v20.9.0/posts/ast_primer.md#scope
        self.root = build_scope(self.node)

    def _reset_root(self):
        self.root = build_scope(self.node)

    def set_node(self, node):
        self.node = node
        self._reset_root()

    def abstracted_table_selects(self) -> Generator[Tuple[str, str], None, None]:
        """For each table in a given query, generates a `SELECT *` query where all unneeded predicates
        are set to `TRUE`.
        We say `unneeded` in the sense that to minimize the data that gets passed to an ingredient,
        we don't need to factor in this operation at the moment.

        Args:
            node: exp.Select node from which to construct abstracted versions of queries for each table.

        Returns:
            abstracted_queries: Generator with (tablename, exp.Select, alias_to_table). The exp.Select is the abstracted query.

        Examples:
            ```python
            scm = SubqueryContextManager(
                node=_parse_one(
                    "SELECT * FROM transactions WHERE {{Model('is this an italian restaurant?', 'transactions::merchant')}} = TRUE AND child_category = 'Restaurants & Dining'"
                )
            )
            scm.abstracted_table_selects()
            ```
            Returns:
            ```text
            ('transactions', 'SELECT * FROM transactions WHERE TRUE AND child_category = \'Restaurants & Dining\'')
            ```
        """
        # TODO: don't really know how to optimize with 'CASE' queries right now
        if self.node.find(exp.Case):
            return
        # Special condition: If...
        #   1) We have a `JOIN` clause
        #   2) We *only* have an ingredient in the top-level `SELECT` clause
        #   3) Our ingredients *only* call a single table
        # ... then we should execute entire rest of SQL first and assign to temporary session table.
        # Example: """SELECT w.title, w."designer ( s )", {{LLMMap('How many animals are in this image?', 'images::title')}}
        #         FROM images JOIN w ON w.title = images.title
        #         WHERE "designer ( s )" = 'georgia gerber'"""
        join_exp = self.node.find(exp.Join)
        if join_exp is not None:
            select_exps = list(self.node.find_all(exp.Select))
            if len(select_exps) == 1:
                # Check if the only `STRUCT` nodes are found in select
                all_struct_exps = list(self.node.find_all(exp.Struct))
                if len(all_struct_exps) > 0:
                    num_select_struct_exps = sum(
                        [
                            len(list(n.find_all(exp.Struct)))
                            for n in select_exps[0].find_all(exp.Alias)
                        ]
                    )
                    if num_select_struct_exps == len(all_struct_exps):
                        if len(self.tables_in_ingredients) == 1:
                            tablename = next(iter(self.tables_in_ingredients))
                            join_tablename = set(
                                [i.name for i in self.node.find_all(exp.Table)]
                            ).difference({tablename})
                            if len(join_tablename) == 1:
                                join_tablename = next(iter(join_tablename))
                                base_select_str = f'SELECT "{tablename}".* FROM "{tablename}", {join_tablename} WHERE '
                                table_conditions_str = self.get_table_predicates_str(
                                    tablename=tablename,
                                    disambiguate_multi_tables=False,
                                )
                                abstracted_query = _parse_one(
                                    base_select_str + table_conditions_str
                                )
                                abstracted_query_str = recover_blendsql(
                                    abstracted_query.sql(dialect=FTS5SQLite)
                                )
                                yield (tablename, abstracted_query_str)
                                return
        for tablename, table_star_query in self._table_star_queries():
            # If this table_star_query doesn't have an ingredient at the top-level, we can safely ignore
            if (
                len(
                    list(
                        get_scope_nodes(
                            root=self.root, nodetype=exp.Struct, restrict_scope=True
                        )
                    )
                )
                == 0
            ):
                continue
            # If our previous subquery has an ingredient, we can't optimize with subquery condition
            # So, remove this subquery constraint and run
            if self.prev_subquery_has_ingredient:
                table_star_query = table_star_query.transform(
                    maybe_set_subqueries_to_true
                )
            # Substitute all ingredients with 'TRUE'
            abstracted_query = table_star_query.transform(set_structs_to_true)
            # Check here to see if we have no other predicates other than 'WHERE TRUE'
            # There's no point in creating a temporary table in this situation
            where_node = abstracted_query.find(exp.Where)
            if where_node:
                if where_node.args["this"] == exp.true():
                    continue
                elif isinstance(where_node.args["this"], exp.Column):
                    continue
                elif all_terminals_are_true(where_node):
                    continue
            elif not where_node:
                continue
            abstracted_query_str = recover_blendsql(
                abstracted_query.sql(dialect=FTS5SQLite)
            )
            yield (tablename, abstracted_query_str)

    def _table_star_queries(
        self,
    ) -> Generator[Tuple[str, exp.Select], None, None]:
        """For each table in the select query, generates a new query
            selecting all columns with the given predicates (Relationships like x = y, x > 1, x >= y).

        Args:
            node: The exp.Select node containing the query to extract table_star queries for

        Returns:
            table_star_queries: Generator with (tablename, exp.Select). The exp.Select is the table_star query

        Examples:
            ```sql
            SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
                FROM account_history
                LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
                WHERE constituents.Sector = 'Information Technology'
                AND lower(Action) like "%dividend%"
            ```
            Returns (after getting str representation of `exp.Select`):
            ```text
            ('account_history', 'SELECT * FROM account_history WHERE lower(Action) like "%dividend%')
            ('constituents', 'SELECT * FROM constituents WHERE sector = \'Information Technology\'')
            ```
        """
        # Use `scope` to get all unique tablenodes in ast
        tablenodes = set(
            list(
                get_scope_nodes(nodetype=exp.Table, root=self.root, restrict_scope=True)
            )
        )
        # aliasnodes catch instances where we do something like
        #   `SELECT (SELECT * FROM x) AS w`
        curr_alias_to_tablename = {}
        curr_alias_to_subquery = {}
        subquery_node = self.node.find(exp.Subquery)
        if subquery_node is not None:
            # Make a note here: we need to create a new table with the name of the alias,
            #   and set to results of this subquery
            alias = None
            if "alias" in subquery_node.args:
                alias = subquery_node.args["alias"]
            if alias is None:
                # Try to get from parent
                parent_node = subquery_node.parent
                if parent_node is not None:
                    if "alias" in parent_node.args:
                        alias = parent_node.args["alias"]
            if alias is not None:
                if not any(x.name == alias.name for x in tablenodes):
                    tablenodes.add(exp.Table(this=exp.Identifier(this=alias.name)))
                curr_alias_to_subquery = {alias.name: subquery_node.args["this"]}
        for tablenode in tablenodes:
            # Check to be sure this is in the top-level `SELECT`
            if is_in_subquery(tablenode):
                continue
            # Check to see if we have a table alias
            # e.g. `SELECT a FROM table AS w`
            table_alias_node = tablenode.find(exp.TableAlias)
            if table_alias_node is not None:
                tablename_to_extract = table_alias_node.name
                curr_alias_to_tablename = {tablename_to_extract: tablenode.name}
                base_select_str = f'SELECT * FROM "{tablenode.name}" AS "{tablename_to_extract}" WHERE '
            else:
                tablename_to_extract = tablenode.name
                base_select_str = f'SELECT * FROM "{tablenode.name}" WHERE '
            table_conditions_str = self.get_table_predicates_str(
                tablename=tablename_to_extract,
                disambiguate_multi_tables=bool(len(tablenodes) > 1)
                or (table_alias_node is not None),
            )
            self.alias_to_tablename = self.alias_to_tablename | curr_alias_to_tablename
            self.tablename_to_alias = self.tablename_to_alias | {
                v: k for k, v in curr_alias_to_tablename.items()
            }
            self.alias_to_subquery = self.alias_to_subquery | curr_alias_to_subquery
            if table_conditions_str:
                yield (
                    tablenode.name,
                    _parse_one(base_select_str + table_conditions_str),
                )

    def get_table_predicates_str(
        self, tablename, disambiguate_multi_tables: bool
    ) -> str:
        """Returns str containing all predicates acting on a specific tablename.

        Args:
            tablename: The target tablename to search and extract predicates for
            disambiguate_multi_tables: `True` if we have multiple tables in our subquery,
                and need to be sure we're only fetching the predicates for the specified `tablename`
        """
        # 2 places conditions can come in here
        # 'WHERE' statement and predicate in a 'JOIN' statement
        all_table_predicates = []
        for table_predicates in get_scope_nodes(
            nodetype=exp.Predicate, root=self.root, restrict_scope=True
        ):
            if is_in_subquery(table_predicates):
                continue
            if disambiguate_multi_tables:
                table_predicates = table_predicates.transform(
                    extract_multi_table_predicates, tablename=tablename
                )
            if isinstance(table_predicates, exp.Expression):
                all_table_predicates.append(table_predicates)
        if len(all_table_predicates) == 0:
            return ""
        table_conditions_str = " AND ".join(
            [c.sql(dialect=FTS5SQLite) for c in all_table_predicates]
        )
        return table_conditions_str

    def infer_gen_constraints(self, start: int, end: int) -> dict:
        """Given syntax of BlendSQL query, infers a regex pattern (if possible) to guide
            downstream Model generations.

        For example:

        ```sql
        SELECT * FROM w WHERE {{LLMMap('Is this true?', 'w::colname')}}
        ```

        We can infer given the structure above that we expect `LLMMap` to return a boolean.
        This function identifies that.

        Arguments:
            indices: The string indices pointing to the span within the overall BlendSQL query
                containing our ingredient in question.

        Returns:
            dict, with keys:

                - output_type
                    - 'boolean' | 'integer' | 'float' | 'string'

                - pattern: regular expression pattern lambda to use in constrained decoding with Model
                    - See `create_pattern` for more info on these pattern lambdas

                - options: Optional str default to pass to `options` argument in a QAIngredient
                    - Will have the form '{table}::{column}'
        """

        def create_pattern(
            output_type: Literal["boolean", "integer", "float"]
        ) -> Callable[[int], str]:
            """Helper function to create a pattern lambda.
            These pattern lambdas take an integer (num_repeats) and return
            a regex pattern which is restricted to repeat exclusively num_repeats times.
            """
            if output_type == "boolean":
                base_pattern = f"((t|f|{DEFAULT_NAN_ANS}){DEFAULT_ANS_SEP})"
            elif output_type == "integer":
                # SQLite max is 18446744073709551615
                # This is 20 digits long, so to be safe, cap the generation at 19
                base_pattern = "((\d{1,18}" + f"|{DEFAULT_NAN_ANS}){DEFAULT_ANS_SEP})"
            elif output_type == "float":
                base_pattern = f"(((\d|\.)+|{DEFAULT_NAN_ANS}){DEFAULT_ANS_SEP})"
            else:
                raise ValueError(f"Unknown output_type {output_type}")
            return lambda num_repeats: base_pattern + "{" + str(num_repeats) + "}"

        added_kwargs: Dict[str, Any] = {}
        ingredient_node = _parse_one(self.sql()[start:end])
        child = None
        for child, _, _ in self.node.walk():
            if child == ingredient_node:
                break
        if child is None:
            raise ValueError
        ingredient_node_in_context = child
        start_node = ingredient_node_in_context.parent
        # Below handles when we're in a function
        # Example: CAST({{LLMMap('jump distance', 'w::notes')}} AS FLOAT)
        while isinstance(start_node, exp.Func) and start_node is not None:
            start_node = start_node.parent
        output_type: Literal["boolean", "integer", "float"] = None
        predicate_literals: List[str] = []
        if start_node is not None:
            predicate_literals = get_predicate_literals(start_node)
            # Check for instances like `{column} = {QAIngredient}`
            # where we can infer the space of possible options for QAIngredient
            if isinstance(start_node, exp.EQ):
                if isinstance(start_node.args["this"], exp.Column):
                    if "table" not in start_node.args["this"].args:
                        logger.debug(
                            "When inferring `options` in infer_gen_kwargs, encountered a column node with "
                            "no table specified!\nShould probably mark `schema_qualify` arg as True"
                        )
                    else:
                        # This is valid for a default `options` set
                        added_kwargs[
                            "options"
                        ] = f"{start_node.args['this'].args['table'].name}::{start_node.args['this'].args['this'].name}"
        if len(predicate_literals) > 0:
            if all(isinstance(x, bool) for x in predicate_literals):
                output_type = "boolean"
            elif all(isinstance(x, float) for x in predicate_literals):
                output_type = "float"
            elif all(isinstance(x, int) for x in predicate_literals):
                output_type = "integer"
            else:
                predicate_literals = [str(i) for i in predicate_literals]
                added_kwargs["output_type"] = "string"
                if len(predicate_literals) > 1:
                    added_kwargs["example_outputs"] = DEFAULT_ANS_SEP.join(
                        predicate_literals
                    )
                else:
                    added_kwargs[
                        "example_outputs"
                    ] = f"{predicate_literals[0]}{DEFAULT_ANS_SEP}{DEFAULT_NAN_ANS}"
                return added_kwargs
        elif isinstance(
            ingredient_node_in_context.parent, (exp.Order, exp.Ordered, exp.AggFunc)
        ):
            output_type = "float"  # Use 'float' as default numeric pattern, since it's more expressive than 'integer'
        if output_type is not None:
            added_kwargs["output_type"] = output_type
            added_kwargs["pattern"] = create_pattern(output_type)
        return added_kwargs

    def sql(self, dialect: sqlglot.dialects.Dialect = FTS5SQLite):
        return recover_blendsql(self.node.sql(dialect=dialect))
