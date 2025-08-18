import sqlglot
from sqlglot import exp, Schema
from sqlglot.optimizer.scope import build_scope
import typing as t
from ast import literal_eval
from sqlglot.optimizer.scope import find_all_in_scope
from attr import attrs, attrib
from colorama import Fore

from blendsql.common.utils import get_tablename_colname
from blendsql.common.constants import ColumnRef
from ..types import QuantifierType, DataTypes
from .dialect import _parse_one
from . import checks as check
from . import transforms as transform
from .constants import SUBQUERY_EXP
from .utils import set_select_to
from blendsql.common.logger import logger


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
    nodetype: t.Type[exp.Expression],
    restrict_scope: bool = False,
    root: t.Optional[sqlglot.optimizer.Scope] = None,
    node: t.Optional[exp.Expression] = None,
) -> t.Generator:
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

    dialect: sqlglot.Dialect = attrib()

    node: exp.Expression = attrib(default=None)
    _query: str = attrib(default=None)
    _last_to_string_node: exp.Expression = None

    def parse(self, query, schema: t.Optional[t.Union[dict, Schema]] = None):
        self._query = query
        self.node = _parse_one(query, dialect=self.dialect, schema=schema)

    def to_string(self):
        # Only call `sql` if we need to
        if hash(self.node) != hash(self._last_to_string_node):
            self._query = self.node.sql(dialect=self.dialect)
            self.last_to_string_node = self.node
        return self._query

    def __setattr__(self, name, value):
        self.__dict__[name] = value


@attrs
class SubqueryContextManager:
    dialect: sqlglot.Dialect = attrib()
    node: exp.Select = attrib()
    prev_subquery_has_ingredient: bool = attrib()
    ingredient_alias_to_parsed_dict: dict = attrib()

    # Keep a running log of what aliases we've initialized so far, per subquery
    alias_to_subquery: dict = attrib(default=None)
    alias_to_tablename: dict = attrib(init=False)
    tablename_to_alias: dict = attrib(init=False)
    columns_referenced_by_ingredients: dict = attrib(init=False)
    root: sqlglot.optimizer.scope.Scope = attrib(init=False)

    def __attrs_post_init__(self):
        self.alias_to_tablename = {}
        self.tablename_to_alias = {}
        # https://github.com/tobymao/sqlglot/blob/v20.9.0/posts/ast_primer.md#scope
        self.root = build_scope(self.node)
        self.columns_referenced_by_ingredients = (
            self.get_columns_referenced_by_ingredients(
                self.ingredient_alias_to_parsed_dict
            )
        )

    def _reset_root(self):
        self.root = build_scope(self.node)

    def set_node(self, node):
        self.node = node
        self._reset_root()

    def get_columns_referenced_by_ingredients(
        self, ingredient_alias_to_parsed_dict: dict
    ):
        # TODO: call infer_gen_constraints() first, to populate `options`
        columns_referenced_by_ingredients = {}
        ingredient_aliases = [i.name for i in check.get_ingredient_nodes(self.node)]
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
                kwargs_dict.get("left_on", None),
                kwargs_dict.get("right_on", None),
            }:
                args = raw_arg
                if not isinstance(raw_arg, (tuple, list)):
                    args = [raw_arg]
                for arg in args:
                    if arg is None:
                        continue
                    # If `context` is a subquery, this gets executed on its own later.
                    if isinstance(arg, ColumnRef):
                        tablename, columnname = get_tablename_colname(arg)
                        if tablename not in columns_referenced_by_ingredients:
                            columns_referenced_by_ingredients[tablename] = set()
                        columns_referenced_by_ingredients[tablename].add(columnname)
        return columns_referenced_by_ingredients

    def abstracted_table_selects(
        self,
    ) -> t.Generator[t.Tuple[str, bool, str], None, None]:
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
        # If we don't have an ingredient at the top-level, we can safely ignore
        elif (
            len(
                list(
                    get_scope_nodes(
                        root=self.root,
                        nodetype=exp.BlendSQLFunction,
                        restrict_scope=True,
                    ),
                )
            )
            == 0
        ):
            return

        self._gather_alias_mappings()
        abstracted_query = self.node.transform(transform.set_ingredient_nodes_to_true)
        # Special condition: If we *only* have an ingredient in the top-level `SELECT` clause
        # ... then we should execute entire rest of SQL first and assign to temporary session table.
        # Example: """SELECT w.title, w."designer ( s )", {{LLMMap('How many animals are in this image?', 'images::title')}}
        #         FROM images JOIN w ON w.title = images.title
        #         WHERE "designer ( s )" = 'georgia gerber'"""
        # Below, we need `self.node.find(exp.Table)` in case we get a QAIngredient on its own
        #   E.g. `SELECT A() AS _col_0` cases should be ignored
        if (
            self.node.find(exp.Table)
            and check.ingredients_only_in_top_select(self.node)
            and not check.ingredient_alias_in_query_body(self.node)
        ):
            for (
                tablename,
                columnnames,
            ) in self.columns_referenced_by_ingredients.items():
                yield (
                    self.alias_to_tablename.get(tablename, tablename),
                    self.node.find(exp.Join) is not None,
                    set_select_to(abstracted_query, tablename, columnnames).sql(
                        dialect=self.dialect
                    ),
                )
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
        if where_node and ignore_join:
            if where_node.args["this"] == exp.true():
                return
            elif isinstance(where_node.args["this"], exp.Column):
                return
            elif check.all_terminals_are_true(where_node):
                return
        elif where_node is None and not ignore_join:
            return
        for tablename, columnnames in self.columns_referenced_by_ingredients.items():
            # TODO: execute query once, and then separate out the results to their respective tables
            yield (
                self.alias_to_tablename.get(tablename, tablename),
                self.node.find(exp.Join) is not None,
                set_select_to(abstracted_query, tablename, columnnames).sql(
                    dialect=self.dialect
                ),
            )
        return

    def _gather_alias_mappings(
        self,
    ) -> t.Generator[t.Tuple[str, exp.Select], None, None]:
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

    def infer_gen_constraints(self, function_node: exp.Expression) -> dict:
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

                - return_type
                    - 'boolean' | 'integer' | 'float' | 'string'

                - regex: regular expression pattern lambda to use in constrained decoding with Model
                    - See `create_regex` for more info on these regex lambdas

                - options: Optional str default to pass to `options` argument in a QAIngredient
                    - Will have the form '{table}.{column}'
        """
        added_kwargs: t.Dict[str, t.Any] = {}
        if isinstance(function_node.parent, exp.Select):
            # We don't want to traverse up in cases of `SELECT {{A()}} FROM table WHERE x < y`
            parent_node = function_node
        else:
            parent_node = function_node.parent
        predicate_literals: t.List[str] = []
        quantifier: QuantifierType = None
        # Check for instances like `{column} = {QAIngredient}`
        # where we can infer the space of possible options for QAIngredient
        if isinstance(parent_node, (exp.EQ, exp.In)):
            if isinstance(parent_node.args["this"], exp.Column):
                if "table" not in parent_node.args["this"].args:
                    if not isinstance(parent_node, exp.BlendSQLFunction):
                        logger.debug(
                            f"When inferring `options` in infer_gen_kwargs, encountered column node with "
                            "no table specified!\nShould probably mark `schema_qualify` arg as True"
                        )
                else:
                    # This is valid for a default `options` set
                    added_kwargs["options"] = ColumnRef(
                        f"{parent_node.args['this'].args['table'].name}.{parent_node.args['this'].args['this'].name}"
                    )
        if isinstance(parent_node, (exp.In, exp.Tuple, exp.Values)):
            if isinstance(parent_node, (exp.Tuple, exp.Values)):
                added_kwargs["wrap_tuple_in_parentheses"] = False
            if isinstance(parent_node, exp.In):
                # If the ingredient is in the 2nd arg place
                # E.g. not `{{LLMMap()}} IN ('a', 'b')`
                # Only `column IN {{LLMQA()}}`
                # AST will look like:
                # In(
                #     this=Column(
                #         this=Identifier(this=name, quoted=False),
                #         table=Identifier(this=c, quoted=False)),
                #     field=BlendSQLFunction(
                #         this=A))
                field_val = parent_node.args.get("field", None)
                if field_val is not None:
                    if parent_node == field_val:
                        quantifier = "+"
                if isinstance(field_val, exp.BlendSQLFunction):
                    quantifier = "+"
            else:
                quantifier = "+"
        if parent_node is not None and parent_node.expression is not None:
            # Get predicate args
            predicate_literals = []
            # Literals
            predicate_literals.extend(
                [
                    literal_eval(i.this) if not i.is_string else i.this
                    for i in parent_node.expression.find_all(exp.Literal)
                ]
            )
            # Booleans
            predicate_literals.extend(
                [i.args["this"] for i in parent_node.expression.find_all(exp.Boolean)]
            )
        # Try to infer output type given the literals we've been given
        # E.g. {{LLMap()}} IN ('John', 'Parker', 'Adam')
        if len(predicate_literals) > 0:
            logger.debug(
                Fore.LIGHTBLACK_EX
                + f"Extracted predicate literals `{predicate_literals}`"
                + Fore.RESET
            )
            if all(isinstance(x, bool) for x in predicate_literals):
                output_type = DataTypes.BOOL(quantifier)
            elif all(isinstance(x, float) for x in predicate_literals):
                output_type = DataTypes.FLOAT(quantifier)
            elif all(isinstance(x, int) for x in predicate_literals):
                output_type = DataTypes.INT(quantifier)
            else:
                predicate_literals = [str(i) for i in predicate_literals]
                added_kwargs["return_type"] = DataTypes.ANY(quantifier)
                if len(predicate_literals) == 1:
                    predicate_literals = predicate_literals + [predicate_literals[0]]
                added_kwargs["example_outputs"] = predicate_literals
                return added_kwargs
        elif len(predicate_literals) == 0 and isinstance(
            parent_node,
            (
                exp.Order,
                exp.Ordered,
                exp.AggFunc,
                exp.GT,
                exp.GTE,
                exp.LT,
                exp.LTE,
                exp.Sum,
            ),
        ):
            output_type = DataTypes.NUMERIC(
                quantifier
            )  # `Numeric` = `t.Union[int, float]`
        elif quantifier:
            # Fallback to a generic list datatype
            output_type = DataTypes.STR(quantifier)
        else:
            output_type = None
        added_kwargs["return_type"] = output_type
        return added_kwargs

    def sql(self):
        return self.node.sql(dialect=self.dialect)
