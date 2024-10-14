import sqlglot
from sqlglot import exp, Schema
from sqlglot.optimizer.scope import build_scope
from typing import (
    Generator,
    List,
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
from sqlglot.optimizer.scope import find_all_in_scope
from attr import attrs, attrib

from ..utils import recover_blendsql
from .._constants import IngredientKwarg
from ._dialect import _parse_one, FTS5SQLite
from . import _checks as check
from . import _transforms as transform
from ._constants import SUBQUERY_EXP
from ._utils import to_select_star
from .._logger import logger


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
    r = [i for i in node.find_all(SUBQUERY_EXP + (exp.Paren,)) if check.in_cte(i)]
    # Then, add (reversed) other subqueries
    return (
        r
        + [
            i for i in node.find_all(SUBQUERY_EXP + (exp.Paren,)) if not check.in_cte(i)
        ][::-1]
    )


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

    def abstracted_table_selects(self) -> Generator[Tuple[str, bool, str], None, None]:
        """For each table in a given query, generates a `SELECT *` query where all unneeded predicates
        are set to `TRUE`.
        We say `unneeded` in the sense that to minimize the data that gets passed to an ingredient,
        we don't need to factor in this operation at the moment.

        Args:
            node: exp.Select node from which to construct abstracted versions of queries for each table.

        Returns:
            abstracted_queries: Generator with (tablename, postprocess_columns, abstracted_query_str).
                postprocess_columns tells us if we potentially executed a query with a `JOIN`, and need to apply some extra post-processing.

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
            ('transactions', False, 'SELECT * FROM transactions WHERE TRUE AND child_category = \'Restaurants & Dining\'')
            ```
        """
        # TODO: don't really know how to optimize with 'CASE' queries right now
        if self.node.find(exp.Case):
            return
        # Special condition: If...
        #   1) We *only* have an ingredient in the top-level `SELECT` clause
        # ... then we should execute entire rest of SQL first and assign to temporary session table.
        # Example: """SELECT w.title, w."designer ( s )", {{LLMMap('How many animals are in this image?', 'images::title')}}
        #         FROM images JOIN w ON w.title = images.title
        #         WHERE "designer ( s )" = 'georgia gerber'"""
        # Below, we need `self.node.find(exp.Table)` in case we get a QAIngredient on its own
        #   E.g. `SELECT A() AS _col_0` should be ignored
        if (
            self.node.find(exp.Table)
            and check.ingredients_only_in_top_select(self.node)
            and not check.ingredient_alias_in_query_body(self.node)
        ):
            abstracted_query = to_select_star(self.node).transform(
                transform.set_structs_to_true
            )
            abstracted_query_str = recover_blendsql(
                abstracted_query.sql(dialect=FTS5SQLite)
            )
            for tablename in self.tables_in_ingredients:
                yield (tablename, True, abstracted_query_str)
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
                    transform.maybe_set_subqueries_to_true
                )
            # Substitute all ingredients with 'TRUE'
            abstracted_query = table_star_query.transform(transform.set_structs_to_true)
            # Check here to see if we have no other predicates other than 'WHERE TRUE'
            # There's no point in creating a temporary table in this situation
            where_node = abstracted_query.find(exp.Where)
            if where_node:
                if where_node.args["this"] == exp.true():
                    continue
                elif isinstance(where_node.args["this"], exp.Column):
                    continue
                elif check.all_terminals_are_true(where_node):
                    continue
            elif not where_node:
                continue
            abstracted_query_str = recover_blendsql(
                abstracted_query.sql(dialect=FTS5SQLite)
            )
            yield (tablename, False, abstracted_query_str)

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
            if check.in_subquery(tablenode):
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
            # Unary operators like `NOT` get parsed as parents of predicate by sqlglot
            # i.e. `SELECT * FROM w WHERE x IS NOT NULL` -> `SELECT * FROM w WHERE NOT x IS NULL`
            # Since these impact the temporary table creation, we consider them parts of the predicate
            #   and fetch them below.
            if isinstance(table_predicates.parent, exp.Unary):
                table_predicates = table_predicates.parent
            if check.in_subquery(table_predicates):
                continue
            if disambiguate_multi_tables:
                table_predicates = table_predicates.transform(
                    transform.extract_multi_table_predicates, tablename=tablename
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

                - regex: regular expression pattern lambda to use in constrained decoding with Model
                    - See `create_regex` for more info on these regex lambdas

                - options: Optional str default to pass to `options` argument in a QAIngredient
                    - Will have the form '{table}::{column}'
        """

        def create_regex(
            output_type: Literal["boolean", "integer", "float"]
        ) -> Callable[[int], str]:
            """Helper function to create a regex lambda.
            These regex lambdas take an integer (num_repeats) and return
            a regex which is restricted to repeat exclusively num_repeats times.
            """
            if output_type == "boolean":
                base_regex = f"(t|f)"
            elif output_type == "integer":
                # SQLite max is 18446744073709551615
                # This is 20 digits long, so to be safe, cap the generation at 19
                base_regex = r"(\d{1,18})"
            elif output_type == "float":
                base_regex = r"(\d(\d|\.)*)"
            else:
                raise ValueError(f"Unknown output_type {output_type}")
            return base_regex

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
                if len(predicate_literals) == 1:
                    predicate_literals = predicate_literals + [predicate_literals[0]]
                added_kwargs["example_outputs"] = predicate_literals
                return added_kwargs
        elif isinstance(
            ingredient_node_in_context.parent, (exp.Order, exp.Ordered, exp.AggFunc)
        ):
            output_type = "float"  # Use 'float' as default numeric regex, since it's more expressive than 'integer'
        if output_type is not None:
            added_kwargs["output_type"] = output_type
            added_kwargs[IngredientKwarg.REGEX] = create_regex(output_type)
        return added_kwargs

    def sql(self, dialect: sqlglot.dialects.Dialect = FTS5SQLite):
        return recover_blendsql(self.node.sql(dialect=dialect))
