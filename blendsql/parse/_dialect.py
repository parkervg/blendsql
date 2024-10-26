from typing import Dict

import itertools
import sqlglot.dialects
from sqlglot.dialects import SQLite, Postgres, DuckDB
from sqlglot.tokens import TokenType
from sqlglot.schema import Schema
from sqlglot import exp, parse_one, alias
from sqlglot.optimizer import qualify_columns as qc
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.scope import traverse_scope, Scope
from typing import Union, Optional
import string

from blendsql.parse._checks import INGREDIENT_PATTERN, is_ingredient_node

INGREDIENT_TOKEN_TYPE_MAPPING: Dict[str, TokenType] = {
    "{{" + f"{letter}()" + "}}": TokenType.FUNCTION for letter in string.ascii_uppercase
}


def _qualify_outputs(scope: Scope) -> None:
    """Ensure all output columns are aliased"""
    new_selections = []

    for i, (selection, aliased_column) in enumerate(
        itertools.zip_longest(scope.expression.selects, scope.outer_column_list)
    ):
        identifier_node = selection.find(exp.Identifier)
        if isinstance(selection, exp.Subquery) or (
            identifier_node is not None and is_ingredient_node(identifier_node)
        ):
            if not selection.output_name:
                selection.set(
                    "alias", exp.TableAlias(this=exp.to_identifier(f"_col_{i}"))
                )
        elif not isinstance(selection, exp.Alias) and not selection.is_star:
            selection = alias(
                selection,
                alias=selection.output_name or f"_col_{i}",
            )
        if aliased_column:
            selection.set("alias", exp.to_identifier(aliased_column))

        new_selections.append(selection)

    scope.expression.set("expressions", new_selections)


def _qualify_columns(scope: Scope, resolver: qc.Resolver) -> None:
    """Disambiguate columns, ensuring each column specifies a source"""
    for column in scope.columns:
        column_table = column.table
        column_name = column.name
        if INGREDIENT_PATTERN.match(column_name) is not None:
            continue

        if column_table and column_table in scope.sources:
            source_columns = resolver.get_source_columns(column_table)
            if (
                source_columns
                and column_name not in source_columns
                and "*" not in source_columns
            ):
                raise OptimizeError(f"Unknown column: {column_name}")

        if not column_table:
            if scope.pivots and not column.find_ancestor(exp.Pivot):
                # If the column is under the Pivot expression, we need to qualify it
                # using the name of the pivoted source instead of the pivot's alias
                column.set("table", exp.to_identifier(scope.pivots[0].alias))
                continue

            column_table = resolver.get_table(column_name)

            # column_table can be a '' because bigquery unnest has no table alias
            if column_table:
                column.set("table", column_table)
        elif column_table not in scope.sources and (
            not scope.parent or column_table not in scope.parent.sources
        ):
            # structs are used like tables (e.g. "struct"."field"), so they need to be qualified
            # separately and represented as dot(dot(...(<table>.<column>, field1), field2, ...))

            root, *parts = column.parts

            if root.name in scope.sources:
                # struct is already qualified, but we still need to change the AST representation
                column_table = root
                root, *parts = parts
            else:
                column_table = resolver.get_table(root.name)

            if column_table:
                column.replace(
                    exp.Dot.build([exp.column(root, table=column_table), *parts])
                )

    for pivot in scope.pivots:
        for column in pivot.find_all(exp.Column):
            if INGREDIENT_PATTERN.match(column.name) is not None:
                continue
            if not column.table and column.name in resolver.all_columns:
                column_table = resolver.get_table(column.name)
                if column_table:
                    column.set("table", column_table)


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

        _qualify_columns(scope, resolver)

        if not schema.empty and expand_alias_refs:
            qc._expand_alias_refs(scope, resolver)

        if not isinstance(scope.expression, exp.UDTF):
            _qualify_outputs(scope)

        qc._expand_group_by(scope)
        qc._expand_order_by(scope, resolver)

    return expression


def glob_to_match(self: SQLite.Generator, expression: exp.Where) -> str:
    return f"{expression.this.sql(dialect=BlendSQLSQLite)} MATCH {expression.expression.sql(dialect=BlendSQLSQLite)}"


class BlendSQLDuckDB(DuckDB):
    class Tokenizer(DuckDB.Tokenizer):
        KEYWORDS = {
            **{
                **DuckDB.Tokenizer.KEYWORDS,
            },
            **INGREDIENT_TOKEN_TYPE_MAPPING,
        }


class BlendSQLPostgres(Postgres):
    class Tokenizer(Postgres.Tokenizer):
        KEYWORDS = {
            **{
                **Postgres.Tokenizer.KEYWORDS,
            },
            **INGREDIENT_TOKEN_TYPE_MAPPING,
        }


class BlendSQLSQLite(SQLite):
    class Tokenizer(SQLite.Tokenizer):
        KEYWORDS = {
            **{
                **SQLite.Tokenizer.KEYWORDS,
                "MATCH": TokenType.GLOB,
            },
            **INGREDIENT_TOKEN_TYPE_MAPPING,
        }

    class Generator(SQLite.Generator):
        TRANSFORMS = {
            **SQLite.Generator.TRANSFORMS,
            exp.Glob: glob_to_match,
        }


def get_dialect(db_type: str) -> sqlglot.dialects.Dialect:
    if db_type == "SQLite":
        return BlendSQLSQLite
    elif db_type == "DuckDB":
        return BlendSQLDuckDB
    elif db_type == "Postgres":
        return BlendSQLPostgres
    else:
        raise ValueError(f"Unknown db_type {db_type}")


def _parse_one(
    sql: str, dialect: sqlglot.Dialect, schema: Optional[Union[dict, Schema]] = None
):
    """Utility to make sure we parse/read queries with the correct dialect."""
    # https://www.sqlite.org/optoverview.html
    node = parse_one(sql, dialect=dialect)
    if schema is not None:
        node = qualify_columns(expression=node, schema=schema, expand_alias_refs=False)
    return node


if __name__ == "__main__":
    e = _parse_one(
        "SELECT * FROM table WHERE {{A()}} > 2", dialect=get_dialect("SQLite")
    )
    # e = _parse_one("SELECT * FROM table WHERE a IN {{A()}}", dialect=get_dialect("DuckDB"))
    print()
