from sqlglot import exp
from dataclasses import dataclass
from typing import Callable
import polars as pl


from blendsql.parse import SubqueryContextManager
from blendsql.common.logger import logger, Color
from blendsql.db import Database
from blendsql.db.utils import double_quote_escape


@dataclass
class BinaryExprContext:
    """Holds the resolved binary expression and related metadata."""

    binary_expr: exp.Binary
    function_node: exp.Exp
    scm: SubqueryContextManager


def find_binary_expression(
    function_node: exp.Exp,
    scm: SubqueryContextManager,
) -> BinaryExprContext | None:
    """
    Resolves the function node and finds its associated binary expression.
    Returns None if no valid binary expression can be found.
    """
    function_node = scm.maybe_resolve_aliased_function(function_node)

    binary_expr = None
    if isinstance(function_node.parent.this, (exp.Binary, exp.In)):
        binary_expr = function_node.parent.this
    elif isinstance(function_node.parent, (exp.Binary, exp.In)):
        if len(list(function_node.parent.find_all(exp.BlendSQLFunction))) == 1:
            binary_expr = function_node.parent

    # Validate: binary expr exists and contains our function node
    if binary_expr is None or binary_expr.find(exp.BlendSQLFunction) != function_node:
        return None

    return BinaryExprContext(
        binary_expr=binary_expr,
        function_node=function_node,
        scm=scm,
    )


def _log_no_cascade_condition(function_node: exp.Exp, scm: SubqueryContextManager):
    logger.debug(
        Color.warning(f"Can't find filter cascade condition for context ")
        + Color.light_warning("`" + function_node.parent.sql(dialect=scm.dialect) + "`")
    )


def _log_cascade_error(e: Exception):
    logger.debug(
        Color.warning(f"Cascade filter logic failed with error: ") + Color.error(str(e))
    )


def _log_executing_cascade(sql: str):
    logger.debug(
        Color.update("Executing ")
        + Color.sql(sql, ignore_prefix=True)
        + Color.update(" to get cascade filter...", ignore_prefix=True)
    )


def execute_cascade_filter(
    function_node: exp.Exp,
    scm: SubqueryContextManager,
    transform_fn: Callable[[exp.Exp], exp.Exp],
    execute_fn: Callable[[exp.Binary], pl.LazyFrame],
):
    """
    Generic cascade filter execution.

    Args:
        function_node: The BlendSQL function node in the AST
        scm: SubqueryContextManager for context
        transform_fn: Function to transform the binary expression
        execute_fn: Function that takes the transformed binary expr and returns the result

    Returns:
        The result of execute_fn, or None if cascade filter cannot be applied.
    """
    try:
        context = find_binary_expression(function_node, scm)

        if context is None:
            _log_no_cascade_condition(function_node, scm)
            return None

        transformed_expr = context.binary_expr.transform(transform_fn)
        return execute_fn(transformed_expr)

    except Exception as e:
        _log_cascade_error(e)
        return None


def get_qa_cascade_filter(
    function_node: exp.Exp,
    scm: SubqueryContextManager,
    db: Database,
    function_result: str | int | float | tuple | bool,
):
    def transform_fn(node):
        if isinstance(node, exp.BlendSQLFunction):
            return exp.Literal(this=str(function_result), is_string=False)
        return node

    def execute_fn(transformed_expr: exp.Binary):
        # At this point, we assume we only have 1 table referenced by ingredients
        assert len(scm.stateful_columns_referenced_by_lm_ingredients) == 1

        colnames_to_select = list(
            scm.stateful_columns_referenced_by_lm_ingredients.values()
        )[0]
        tablename = list(scm.stateful_columns_referenced_by_lm_ingredients.keys())[0]
        _tablename = scm.alias_to_tablename.get(tablename, tablename)

        alias_exp = ""
        if _tablename != tablename:
            alias_exp = f"{tablename}"
            tablename = _tablename

        cascade_filter_sql = (
            f"SELECT {', '.join(colnames_to_select)} FROM {tablename} {alias_exp} "
            f"WHERE {transformed_expr.sql()}"
        )
        _log_executing_cascade(cascade_filter_sql)
        return db.execute_to_df(cascade_filter_sql)

    return execute_cascade_filter(function_node, scm, transform_fn, execute_fn)


def get_map_cascade_filter(
    function_node: exp.Exp,
    tablename: str,
    scm: SubqueryContextManager,
    new_col: str,
    new_table: pl.LazyFrame,
) -> pl.LazyFrame | None:
    def transform_fn(node):
        if isinstance(node, exp.BlendSQLFunction):
            return exp.Column(
                this=exp.Identifier(this=double_quote_escape(new_col), quoted=True)
            )
        return node

    def execute_fn(transformed_expr: exp.Binary):
        cascade_filter_sql = (
            f"SELECT * FROM self AS {tablename} " f"WHERE {transformed_expr.sql()}"
        )
        _log_executing_cascade(cascade_filter_sql)

        colnames_to_select = scm.stateful_columns_referenced_by_lm_ingredients[
            scm.tablename_to_alias.get(tablename, tablename)
        ]
        return new_table.sql(cascade_filter_sql).select(
            [
                pl.col(col)
                for col in colnames_to_select
                if col in new_table.collect_schema().names()
            ]
        )

    return execute_cascade_filter(function_node, scm, transform_fn, execute_fn)
