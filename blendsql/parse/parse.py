import sqlglot
from sqlglot import exp, Schema
from sqlglot.optimizer.scope import build_scope
from typing import Callable, Type, Generator
import re
from collections import Counter
from sqlglot.optimizer.scope import find_all_in_scope
from dataclasses import dataclass, field
from functools import partial

from blendsql.common.utils import get_tablename_colname
from blendsql.common.typing import ColumnRef, StringConcatenation
from blendsql.common.logger import logger, Color
from blendsql.db import Database
from blendsql.parse.dialect import _parse_one
from blendsql.parse import checks as check
from blendsql.parse import transforms as transform
from blendsql.parse.utils import set_select_to
from blendsql.parse.return_type_inferrer import ReturnTypeInferrer


def get_reversed_subqueries(node):
    """Iterates through all subqueries (either parentheses or select).
    Reverses all EXCEPT for CTEs, which should remain in order.
    """
    # First, fetch all common table expressions
    r = [
        i
        for i in node.find_all(
            (
                exp.Select,
                exp.Paren,
            )
        )
        if check.in_cte(i)
    ]
    # Then, add (reversed) other subqueries
    return (
        r
        + [
            i
            for i in node.find_all(
                (
                    exp.Select,
                    exp.Paren,
                )
            )
            if not check.in_cte(i)
        ][::-1]
    )


def get_scope_nodes(
    nodetype: Type[exp.Expression],
    restrict_scope: bool = False,
    root: sqlglot.optimizer.Scope | None = None,
    node: exp.Expression | None = None,
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


@dataclass
class QueryContextManager:
    dialect: sqlglot.Dialect = field()
    node: exp.Expression = field(default=None)

    def parse(self, query: str, schema: dict | Schema | None = None):
        self.node = _parse_one(query, dialect=self.dialect, schema=schema)

    def to_string(self):
        return self.node.sql(dialect=self.dialect)


@dataclass
class SubqueryContextManager:
    dialect: sqlglot.Dialect = field()
    node: exp.Select = field()
    prev_subquery_has_ingredient: bool = field()
    ingredient_alias_to_parsed_dict: dict = field()

    # Keep a running log of what aliases we've initialized so far, per subquery
    alias_to_subquery: dict = field(default_factory=dict)
    alias_to_tablename: dict = field(default_factory=dict)
    tablename_to_alias: dict = field(default_factory=dict)

    root: sqlglot.optimizer.scope.Scope = field(init=False)
    stateful_columns_referenced_by_lm_ingredients: dict = field(init=False)
    function_references: list[exp.Expression] = field(init=False)

    def __post_init__(self):
        self.alias_to_tablename = dict()
        self.tablename_to_alias = dict()
        self.self_join_tablenames = set()
        # https://github.com/tobymao/sqlglot/blob/v20.9.0/posts/ast_primer.md#scope
        self.root = build_scope(self.node)
        self.stateful_columns_referenced_by_lm_ingredients = (
            self.get_stateful_columns_referenced_by_lm_functions(
                self.ingredient_alias_to_parsed_dict
            )
        )
        self.function_references = list(self.node.find_all(exp.BlendSQLFunction))
        self._gather_alias_mappings()
        self.return_type_inferrer = ReturnTypeInferrer()

    def _reset_root(self):
        self.root = build_scope(self.node)

    def set_node(self, node):
        self.node = node
        self._reset_root()

    def get_stateful_columns_referenced_by_lm_functions(
        self, ingredient_alias_to_parsed_dict: dict
    ):
        stateful_columns_referenced_by_lm_functions = {}
        ingredient_aliases = [i.name for i in check.get_ingredient_nodes(self.node)]

        def _process_single(arg: ColumnRef):
            tablename, columnname = get_tablename_colname(arg)
            if tablename not in stateful_columns_referenced_by_lm_functions:
                stateful_columns_referenced_by_lm_functions[tablename] = set()
            stateful_columns_referenced_by_lm_functions[tablename].add(columnname)

        for ingredient_alias in ingredient_aliases:
            kwargs_dict = ingredient_alias_to_parsed_dict[ingredient_alias][
                "kwargs_dict"
            ]
            for raw_arg in {
                # Below lists all arguments where a table may be referenced
                # We omit `options`, since this should not take into account the
                #   state of the filtered database.
                kwargs_dict.get("context", None),
                kwargs_dict.get("values", None),
                kwargs_dict.get("additional_args", None),
                kwargs_dict.get("left_on", None),
                kwargs_dict.get("right_on", None),
            }:
                args = raw_arg
                if not isinstance(raw_arg, (tuple, list)):
                    args = [raw_arg]
                for arg in args:
                    if arg is None:
                        continue
                    if isinstance(arg, StringConcatenation):
                        for column in arg:
                            _process_single(column)
                    elif isinstance(arg, ColumnRef):
                        _process_single(arg)
                    # If `context` is a subquery, this gets executed on its own later, so we don't handle it here.
        return stateful_columns_referenced_by_lm_functions

    def abstracted_table_selects(
        self, db: Database
    ) -> Generator[tuple[str, str], None, None]:
        """For each table in a given query, generates a `SELECT *` query where all unneeded predicates
        are set to `TRUE`.
        We say `unneeded` in the sense that to minimize the data that gets passed to an ingredient,
        we don't need to factor in this operation at the moment.

        Args:
            node: exp.Select node from which to construct abstracted versions of queries for each table.

        Returns:
            abstracted_queries: Generator with (tablename, abstracted_query_str).

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
        # If we don't have an ingredient at the top-level, we can safely ignore
        if len(self.stateful_columns_referenced_by_lm_ingredients) == 0:
            return

        abstracted_query = self.node.transform(transform.set_ingredient_nodes_to_true)

        # Prepare join metadata if multiple tables are referenced
        abstracted_join_temp_tablename = None
        all_tablename_or_aliasnames = []
        all_columnnames = []

        if len(self.stateful_columns_referenced_by_lm_ingredients) > 1:
            all_resolved_tablenames = []
            for (
                tablename_or_aliasname,
                columnnames,
            ) in self.stateful_columns_referenced_by_lm_ingredients.items():
                tablename = self.alias_to_tablename.get(
                    tablename_or_aliasname, tablename_or_aliasname
                )
                all_resolved_tablenames.append(tablename)
                columnnames = list(columnnames)
                all_tablename_or_aliasnames.extend(
                    [tablename_or_aliasname] * len(columnnames)
                )
                all_columnnames.extend(columnnames)

            abstracted_join_temp_tablename = "_JOIN_".join(all_resolved_tablenames)

        def prepare_joined_temp_table():
            abstracted_join_str = set_select_to(
                node=abstracted_query,
                tablenames=all_tablename_or_aliasnames,
                columnnames=all_columnnames,
                aliasnames=[
                    f"{c}_{t}"
                    for c, t in zip(all_columnnames, all_tablename_or_aliasnames)
                ],
            ).sql(dialect=self.dialect)
            logger.debug(
                Color.update("Executing ")
                + Color.sql(abstracted_join_str, ignore_prefix=True)
                + Color.update(
                    f" and setting to `{abstracted_join_temp_tablename}`...",
                    ignore_prefix=True,
                )
            )
            abstracted_join_df = db.execute_to_df(abstracted_join_str)
            db.to_temp_table(
                df=abstracted_join_df, tablename=abstracted_join_temp_tablename
            )

        def _result(abstracted_query, tablename_or_aliasname, columnnames):
            resolved_tablename = self.alias_to_tablename.get(
                tablename_or_aliasname, tablename_or_aliasname
            )
            if abstracted_join_temp_tablename is not None:
                query = set_select_to(
                    exp.Select(
                        expressions=[exp.Star()],
                        from_=exp.From(
                            this=exp.Table(
                                this=exp.Identifier(this=abstracted_join_temp_tablename)
                            )
                        ),
                    ),
                    tablenames=[abstracted_join_temp_tablename] * len(columnnames),
                    columnnames=[f"{c}_{tablename_or_aliasname}" for c in columnnames],
                    aliasnames=list(columnnames),
                )
            else:
                query = set_select_to(
                    abstracted_query,
                    [tablename_or_aliasname] * len(columnnames),
                    list(columnnames),
                )
            return (resolved_tablename, has_join, query.sql(dialect=self.dialect))

        has_join = self.node.find(exp.Join) is not None

        # Special condition: If we *only* have an ingredient in the top-level `SELECT` clause,
        #   then we can be more aggressive and execute the ENTIRE rest of SQL first and assign to temporary session table.
        # Example: """SELECT w.title, w."designer ( s )", {{LLMMap('How many animals are in this image?', 'images::title')}}
        #    FROM images JOIN w ON w.title = images.title
        #    WHERE "designer ( s )" = 'georgia gerber'"""
        # Below, we also need `self.node.find(exp.Table)` in case we get a QAIngredient on its own
        #   E.g. `SELECT A() AS _col_0` cases should be ignored
        if (
            self.node.find(exp.Table)
            and check.ingredients_only_in_top_select(self.node)
            and not check.ingredient_alias_in_query_body(self.node)
        ):
            if abstracted_join_temp_tablename is not None:
                prepare_joined_temp_table()
            for (
                tablename_or_aliasname,
                columnnames,
            ) in self.stateful_columns_referenced_by_lm_ingredients.items():
                yield _result(abstracted_query, tablename_or_aliasname, columnnames)
            return

        # Base case is below
        abstracted_query = abstracted_query.transform(
            transform.remove_nodetype,
            (exp.Order, exp.Limit, exp.Group, exp.Offset, exp.Having),
        )
        # If our previous subquery has an ingredient, we can't optimize with subquery condition
        # So, remove this subquery constraint and run
        if self.prev_subquery_has_ingredient:
            abstracted_query = abstracted_query.transform(
                transform.maybe_set_subqueries_to_true
            )
        # Happens with {{LLMQA()}} cases, where we get 'SELECT *'
        if abstracted_query.find(exp.Table) is None:
            return
        # Check here to see if we have no other predicates other than 'WHERE TRUE'
        # There's no point in creating a temporary table in this situation
        where_node = abstracted_query.find(exp.Where)
        join_node = abstracted_query.find(exp.Join)
        # If we have a join_node that's a cross join ('JOIN "colors" ON TRUE'),
        #   this was likely created by a LLMJoin ingredient.
        #   We don't need to create temp tables for these.
        # TODO: This cross join is inefficient, make it a union
        is_cross_join = lambda node: node.args.get("on", None) == exp.true()
        ignore_join = bool(not join_node or is_cross_join(join_node))

        if ignore_join and where_node:
            where_this = where_node.args["this"]
            if (
                where_this == exp.true()
                or isinstance(where_this, exp.Column)
                or check.all_terminals_are_true(where_node)
            ):
                return
        elif not ignore_join and where_node is None:
            return

        if abstracted_join_temp_tablename is not None:
            prepare_joined_temp_table()

        for (
            tablename_or_aliasname,
            columnnames,
        ) in self.stateful_columns_referenced_by_lm_ingredients.items():
            yield _result(abstracted_query, tablename_or_aliasname, columnnames)
        return

    def _gather_alias_mappings(
        self,
    ) -> Generator[tuple[str, exp.Select], None, None]:
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
            if check.in_subquery(tablenode):
                continue
            # Check to see if we have a table alias
            # e.g. `SELECT a FROM table AS w`
            table_alias_node = tablenode.find(exp.TableAlias)
            if table_alias_node is not None:
                curr_alias_to_tablename = {table_alias_node.name: tablenode.name}
            self.alias_to_tablename |= curr_alias_to_tablename
            self.tablename_to_alias |= {
                v: k for k, v in curr_alias_to_tablename.items()
            }

            self.alias_to_subquery |= curr_alias_to_subquery
        self.self_join_tablenames.update(
            [
                table
                for table, count in Counter(self.alias_to_tablename.values()).items()
                if count > 1
            ]
        )

    def maybe_resolve_aliased_function(
        self, function_node: exp.Expression
    ) -> exp.Expression:
        """More specifically, this function takes an exp.BlendSQLFunction, and returns an exp.BlendSQLFunction."""
        # is this function_node an alias in a `SELECT` statement?
        if isinstance(function_node.parent, exp.Alias):
            for _node in self.function_references:
                if (
                    isinstance(_node.parent, exp.Binary)
                    and _node.this == function_node.this
                ):
                    return _node
                    # TODO: this can be made more robust by finding ALL references of this function,
                    #   and seeing if we can combine them into a single exit_condition.
        return function_node

    def get_exit_condition(self, function_node: exp.Expression) -> Callable | None:
        limit_node = self.node.find(exp.Limit)
        if limit_node is None:
            return None

        def _has_unsafe_or(node):
            """
            An OR is unsafe if it's not contained within parentheses
            at the WHERE clause level.
            """
            for or_node in node.find_all(exp.Or):
                # Check if this OR has a Paren as an ancestor before hitting WHERE/AND
                parent = or_node.parent
                while parent and parent != node:
                    if isinstance(parent, exp.Paren):
                        break
                    if isinstance(parent, (exp.Where, exp.And)):
                        return True
                    parent = parent.parent
            return False

        # First, check for expressions that always invalidate early exit
        if (
            self.node.find(exp.Group)
            or self.node.find(exp.Order)
            or self.node.find(exp.Distinct)
        ):
            return None

        # For `OR`, we need to check if it appears at the "top level"
        # vs being contained within a subexpression
        # `... {{A()}} AND (x OR y) LIMIT 5` should still be eligible for an exit condition
        where_clause = self.node.find(exp.Where)
        if where_clause and _has_unsafe_or(where_clause):
            return None

        function_node = self.maybe_resolve_aliased_function(function_node)

        if isinstance(function_node.parent, (exp.Binary, exp.In)):
            # We can apply some exit_condition function
            limit_arg: int = limit_node.expression.to_py()
            offset_node = self.node.find(exp.Offset)
            offset_arg = 0
            if offset_node:
                offset_arg: int = offset_node.expression.to_py()
            parent_node = function_node.parent
            arg = parent_node.expression.to_py()
            _exit_condition = (
                lambda d, op: sum(op(v) for v in d.values()) >= limit_arg + offset_arg
            )
            if arg == function_node:
                return None
            if isinstance(parent_node, exp.EQ):
                return partial(_exit_condition, op=lambda v: v == arg)
            elif isinstance(parent_node, exp.GT):
                return partial(_exit_condition, op=lambda v: v > arg)
            elif isinstance(parent_node, exp.GTE):
                return partial(_exit_condition, op=lambda v: v >= arg)
            elif isinstance(parent_node, exp.LT):
                return partial(_exit_condition, op=lambda v: v < arg)
            elif isinstance(parent_node, exp.LTE):
                return partial(_exit_condition, op=lambda v: v <= arg)
            elif isinstance(parent_node, exp.Like):
                # First we need to convert SQL pattern to regex
                re_pattern = re.escape(arg).replace(r"\%", ".*")
                return partial(_exit_condition, op=lambda v: re.search(re_pattern, v))
            elif isinstance(parent_node, exp.Is):
                return partial(_exit_condition, op=lambda v: v is arg)
            elif isinstance(parent_node, exp.Not):
                return partial(_exit_condition, op=lambda v: not v)
            elif isinstance(parent_node, exp.In):
                print()
            # TODO: add more

    def is_eligible_for_cascade_filter(self) -> bool:
        """
        A query is eligible for cascade filtering if:
        1. It's a single-table query
        2. It has 2+ BlendSQL functions in the WHERE clause
        3. Those functions are not separated by OR operators
        4. There are no BlendSQL functions outside the WHERE clause (not yet supported)
        """

        def has_or_with_blendsql(node):
            """Check if node is/contains OR with BlendSQL functions in different branches"""
            if node is None:
                return False
            if isinstance(node, exp.Or):
                # Check if both sides of OR contain BlendSQL functions
                left_has_blendsql = any(
                    isinstance(n, exp.BlendSQLFunction) for n in node.left.walk()
                )
                right_has_blendsql = any(
                    isinstance(n, exp.BlendSQLFunction) for n in node.right.walk()
                )

                # If BlendSQL functions exist in the OR, cascading is unsafe
                if left_has_blendsql or right_has_blendsql:
                    return True

            # Recursively check children
            for child in node.iter_expressions():
                if has_or_with_blendsql(child):
                    return True
            return False

        where_node = self.node.find(exp.Where)
        if where_node is None:
            return False

        # Count BlendSQL functions in WHERE clause
        blendsql_functions_in_where = [
            n for n in where_node.walk() if isinstance(n, exp.BlendSQLFunction)
        ]

        select_node = self.node.find(exp.Select)
        if select_node is None:
            return False

        # Count BlendSQL functions in WHERE clause
        blendsql_functions_in_select = [
            n for n in select_node.walk() if isinstance(n, exp.BlendSQLFunction)
        ]

        # Need at least 2 functions to cascade
        if len(blendsql_functions_in_where) < 2:
            return False

        # Check if there's an OR that makes cascading unsafe
        if has_or_with_blendsql(where_node):
            return False

        # Check for BlendSQL functions outside WHERE clause
        all_blendsql_functions = [
            n for n in self.node.walk() if isinstance(n, exp.BlendSQLFunction)
        ]

        # if len(set(list(self.node.find_all(exp.Table)))) > 1:
        #     logger.debug(
        #         Color.error(f"Can't apply filter cascade on multi-table queries yet ):")
        #     )
        #     return False

        if len(all_blendsql_functions) > (
            len(blendsql_functions_in_where) + len(blendsql_functions_in_select)
        ):
            logger.debug(
                Color.error(
                    "Cascade filtering optimization is not yet supported for queries with "
                    "BlendSQL functions outside the WHERE or SELECT clause (e.g., in ORDER BY, HAVING, etc.)"
                )
            )
            return False

        return True

    def infer_gen_constraints(
        self,
        function_node: exp.Expression,
        schema: dict,
        alias_to_tablename: dict,
        has_user_regex: bool,
    ) -> dict:
        """
        Convenience function matching the original method signature.

        Args:
            function_node: The expression node containing the BlendSQL function
            schema: Database schema mapping table names to column types
            alias_to_tablename: Mapping of table aliases to actual table names
            has_user_regex: Whether the user has provided a custom regex

        Returns:
            Dict with inferred generation constraints
        """
        return self.return_type_inferrer(
            function_node=self.maybe_resolve_aliased_function(function_node),
            schema=schema,
            alias_to_tablename=alias_to_tablename,
            has_user_regex=has_user_regex,
        )

    def sql(self):
        return self.node.sql(dialect=self.dialect)
