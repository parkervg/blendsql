from sqlglot.dialects import SQLite
from sqlglot.tokens import TokenType
from sqlglot.schema import Schema
from sqlglot import exp, parse_one
from sqlglot.optimizer import qualify_columns as qc
from sqlglot.optimizer.scope import traverse_scope
from typing import Union, Optional


def qualify_columns(
    expression: exp.Expression,
    schema: Union[dict, Schema],
    expand_alias_refs: bool = True,
    infer_schema: Optional[bool] = None,
) -> exp.Expression:
    """
    *******************************************************
    Below is copied from sqlglot.optimizer.qualify_columns.
    We remove the '_expand_stars()' call.
    *******************************************************

    Rewrite sqlglot AST to have fully qualified columns.

    Example:
        >>> import sqlglot
        >>> schema = {"tbl": {"col": "INT"}}
        >>> expression = sqlglot.parse_one("SELECT col FROM tbl")
        >>> qualify_columns(expression, schema).sql()
        'SELECT tbl.col AS col FROM tbl'

    Args:
        expression: Expression to qualify.
        schema: Database schema.
        expand_alias_refs: Whether or not to expand references to aliases.
        infer_schema: Whether or not to infer the schema if missing.

    Returns:
        The qualified expression.
    """

    schema = qc.ensure_schema(schema)
    infer_schema = schema.empty if infer_schema is None else infer_schema

    for scope in traverse_scope(expression):
        resolver = qc.Resolver(scope, schema, infer_schema=infer_schema)
        qc._pop_table_column_aliases(scope.ctes)
        qc._pop_table_column_aliases(scope.derived_tables)

        if schema.empty and expand_alias_refs:
            qc._expand_alias_refs(scope, resolver)

        qc._qualify_columns(scope, resolver)

        if not schema.empty and expand_alias_refs:
            qc._expand_alias_refs(scope, resolver)

        if not isinstance(scope.expression, exp.UDTF):
            qc._qualify_outputs(scope)

        qc._expand_group_by(scope)
        qc._expand_order_by(scope, resolver)

    return expression


def glob_to_match(self: SQLite.Generator, expression: exp.Where) -> str:
    return f"{expression.this.sql(dialect=FTS5SQLite)} MATCH {expression.expression.sql(dialect=FTS5SQLite)}"


class FTS5SQLite(SQLite):
    """SQLite dialect extended to handle the builtin FTS5 extension.
    https://www.sqlite.org/fts5.html
    Essentially, all we do here is set `GLOB` to render as `MATCH`
    """

    class Tokenizer(SQLite.Tokenizer):
        KEYWORDS = {**SQLite.Tokenizer.KEYWORDS, "MATCH": TokenType.GLOB}

    class Generator(SQLite.Generator):
        TRANSFORMS = {
            **SQLite.Generator.TRANSFORMS,
            exp.Glob: glob_to_match,
        }


def _parse_one(sql: str, schema: Optional[Union[dict, Schema]] = None):
    """Utility to make sure we parse/read queries with the correct dialect."""
    # https://www.sqlite.org/optoverview.html
    node = parse_one(sql, dialect=FTS5SQLite)
    if schema is not None:
        node = qualify_columns(expression=node, schema=schema, expand_alias_refs=False)
    return node
