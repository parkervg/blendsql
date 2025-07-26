import typing as t
import sqlglot.dialects
from sqlglot.dialects import SQLite, Postgres, DuckDB
from sqlglot.tokens import TokenType
from sqlglot.schema import Schema, MappingSchema
from sqlglot import exp, parse_one
from typing import Union, Optional
from sqlglot import exp
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.optimizer.isolate_table_selects import isolate_table_selects
from sqlglot.optimizer.qualify_columns import (
    pushdown_cte_alias_columns as pushdown_cte_alias_columns_func,
    qualify_columns as qualify_columns_func,
    quote_identifiers as quote_identifiers_func,
    validate_qualify_columns as validate_qualify_columns_func,
)
from sqlglot.schema import Schema, ensure_schema

import sqlglot
from sqlglot import TokenType
from sqlglot.dialects.duckdb import DuckDB
from sqlglot import expressions as exp

# Use existing TokenType values that are rarely used in SQL
# We'll repurpose BLOCK_START and BLOCK_END for our function brackets
L_FUNC_BRACKET = TokenType.BLOCK_START  # {{
R_FUNC_BRACKET = TokenType.BLOCK_END  # }}


class BlendSQLFunction(exp.Expression):
    """Custom AST node for function calls within {{ }} brackets"""

    arg_types = {"fn_args": False, "fn_kwargs": False}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def fn_args(self):
        """Get the function expression"""
        return self.args.get("fn_args")

    @property
    def fn_kwargs(self):
        """Get the keyword arguments"""
        return self.args.get("fn_kwargs", [])


class KeywordArgument(exp.Expression):
    """Represents a keyword argument like quantifier='{3}'"""

    arg_types = {"name": True, "value": True}

    @property
    def name(self):
        return self.args.get("name")

    @property
    def value(self):
        return self.args.get("value")


class BlendSQLDialect(sqlglot.Dialect):
    class Tokenizer(sqlglot.Tokenizer):
        # Add our custom token mappings using existing TokenType values
        KEYWORDS = {
            **sqlglot.Tokenizer.KEYWORDS,
            "{{": L_FUNC_BRACKET,
            "}}": R_FUNC_BRACKET,
        }

    class Parser(sqlglot.Parser):
        def _parse_primary(self):
            """Override primary parsing to handle our function brackets"""
            # Check if we have a function bracket
            if self._match(L_FUNC_BRACKET):
                return self._parse_function_bracket()

            # Otherwise, use the default primary parsing
            return super()._parse_primary()

        def _parse_function_bracket(self):
            """Parse content within {{ }} specifically as a function call with keyword args"""
            # We've already consumed the {{ token

            # Parse the function name
            func_name = self._parse_id_var()
            if not func_name:
                self.raise_error("Expected function name within {{ }} brackets")

            # Expect opening parenthesis
            if not self._match(TokenType.L_PAREN):
                self.raise_error("Expected '(' after function name")

            # Parse positional arguments and keyword arguments
            args = []
            kwargs = []

            # Parse arguments until we hit the closing parenthesis
            while self._curr and self._curr.token_type != TokenType.R_PAREN:
                # Check if this looks like a keyword argument (identifier followed by =)
                if (
                    # We only need to specify `TokenType.VALUES` here
                    #   because we have a 'values' kwarg. But, I don't think we'll
                    #   be explicitly passing this in future syntax, so we can remove it below.
                    self._curr.token_type in (TokenType.VAR, TokenType.VALUES)
                    and self._index + 1 < len(self._tokens)
                    and self._tokens[self._index + 1].token_type == TokenType.EQ
                ):
                    keyword_name = self._curr.text
                    value = self._parse_conjunction()
                    kwargs.append(
                        KeywordArgument(
                            name=exp.Identifier(this=keyword_name),
                            value=value.expression,
                        )
                    )
                else:
                    # Parse as positional argument
                    arg = self._parse_conjunction()
                    args.append(arg)

                # Handle comma separation
                if self._match(TokenType.COMMA):
                    continue
                elif self._curr and self._curr.token_type == TokenType.R_PAREN:
                    break
                else:
                    self.raise_error("Expected ',' or ')' in function arguments")

            # Consume the closing parenthesis
            if not self._match(TokenType.R_PAREN):
                self.raise_error("Expected ')' to close function arguments")

            # Expect the closing }} bracket
            if not self._match(R_FUNC_BRACKET):
                self.raise_error("Expected '}}' to close function bracket")

            # Create a custom AST node to represent our function bracket
            return BlendSQLFunction(this=func_name.name, fn_args=args, fn_kwargs=kwargs)

    class Generator(sqlglot.Generator):
        def blendsqlfunction_sql(self, expression: BlendSQLFunction) -> str:
            """Generate SQL for BlendSQLFunction nodes"""
            func_sql = f"{expression.name}("
            arg_sql, kwarg_sql = None, None
            if expression.fn_args is not None:
                arg_sql = ", ".join([self.sql(arg) for arg in expression.fn_args])
            if expression.fn_kwargs is not None:
                kwarg_sql = ", ".join(
                    [self.sql(kwarg) for kwarg in expression.fn_kwargs]
                )
            combined = [i for i in [arg_sql, kwarg_sql] if i]
            func_sql += f"{', '.join(combined)})"
            return f"{{{{{func_sql}}}}}"

        def keywordargument_sql(self, expression: KeywordArgument) -> str:
            """Generate SQL for KeywordArgument nodes"""
            name_sql = self.sql(expression.name)
            value_sql = self.sql(expression.value)
            return f"{name_sql}={value_sql}"


class BlendSQLDuckDB(BlendSQLDialect, DuckDB):
    pass


class BlendSQLPostgres(BlendSQLDialect, Postgres):
    pass


def glob_to_match(self: SQLite.Generator, expression: exp.Where) -> str:
    return f"{expression.this.sql(dialect=BlendSQLSQLite)} MATCH {expression.expression.sql(dialect=BlendSQLSQLite)}"


def str_position_to_substr(self: SQLite.Generator, expression: exp.Where) -> str:
    return f"INSTR({expression.this.sql(dialect=BlendSQLSQLite)}, {expression.args['substr'].sql(dialect=BlendSQLSQLite)})"


class BlendSQLSQLite(BlendSQLDialect, SQLite):
    class Tokenizer(BlendSQLDialect.Tokenizer, SQLite.Tokenizer):
        KEYWORDS = {
            **{
                **BlendSQLDialect.Tokenizer.KEYWORDS,
                "MATCH": TokenType.GLOB,
            },
        }

    class Generator(BlendSQLDialect.Generator, SQLite.Generator):
        TRANSFORMS = {
            **BlendSQLDialect.Generator.TRANSFORMS,
            exp.Glob: glob_to_match,
            exp.StrPosition: str_position_to_substr,
        }


def get_dialect(db_type: str) -> sqlglot.dialects.Dialect:
    if db_type == "SQLite":
        return BlendSQLSQLite
    elif db_type == "DuckDB":
        return BlendSQLDuckDB
    elif db_type == "PostgreSQL":
        return BlendSQLPostgres
    else:
        raise ValueError(f"Unknown db_type {db_type}")


def qualify(
    expression: exp.Expression,
    dialect: DialectType = None,
    db: t.Optional[str] = None,
    catalog: t.Optional[str] = None,
    schema: t.Optional[dict | Schema] = None,
    expand_alias_refs: bool = True,
    expand_stars: bool = True,
    infer_schema: t.Optional[bool] = None,
    isolate_tables: bool = False,
    qualify_columns: bool = True,
    allow_partial_qualification: bool = False,
    validate_qualify_columns: bool = True,
    quote_identifiers: bool = True,
    identify: bool = True,
    infer_csv_schemas: bool = False,
) -> exp.Expression:
    """
    Rewrite sqlglot AST to have normalized and qualified tables and columns.

    This step is necessary for all further SQLGlot optimizations.

    Example:
        >>> import sqlglot
        >>> schema = {"tbl": {"col": "INT"}}
        >>> expression = sqlglot.parse_one("SELECT col FROM tbl")
        >>> qualify(expression, schema=schema).sql()
        'SELECT "tbl"."col" AS "col" FROM "tbl" AS "tbl"'

    Args:
        expression: Expression to qualify.
        db: Default database name for tables.
        catalog: Default catalog name for tables.
        schema: Schema to infer column names and types.
        expand_alias_refs: Whether to expand references to aliases.
        expand_stars: Whether to expand star queries. This is a necessary step
            for most of the optimizer's rules to work; do not set to False unless you
            know what you're doing!
        infer_schema: Whether to infer the schema if missing.
        isolate_tables: Whether to isolate table selects.
        qualify_columns: Whether to qualify columns.
        allow_partial_qualification: Whether to allow partial qualification.
        validate_qualify_columns: Whether to validate columns.
        quote_identifiers: Whether to run the quote_identifiers step.
            This step is necessary to ensure correctness for case sensitive queries.
            But this flag is provided in case this step is performed at a later time.
        identify: If True, quote all identifiers, else only necessary ones.
        infer_csv_schemas: Whether to scan READ_CSV calls in order to infer the CSVs' schemas.

    Returns:
        The qualified expression.
    """
    schema = ensure_schema(schema, dialect=dialect)
    # COMMENTED OUT THE TWO LINES BELOW
    # expression = qualify_tables(
    #     expression,
    #     db=db,
    #     catalog=catalog,
    #     schema=schema,
    #     dialect=dialect,
    #     infer_csv_schemas=infer_csv_schemas,
    # )
    # expression = normalize_identifiers(expression, dialect=dialect)

    if isolate_tables:
        expression = isolate_table_selects(expression, schema=schema)

    if Dialect.get_or_raise(dialect).PREFER_CTE_ALIAS_COLUMN:
        expression = pushdown_cte_alias_columns_func(expression)

    if qualify_columns:
        expression = qualify_columns_func(
            expression,
            schema,
            expand_alias_refs=expand_alias_refs,
            expand_stars=expand_stars,
            infer_schema=infer_schema,
            allow_partial_qualification=allow_partial_qualification,
        )

    if quote_identifiers:
        expression = quote_identifiers_func(
            expression, dialect=dialect, identify=identify
        )

    if validate_qualify_columns:
        validate_qualify_columns_func(expression)

    return expression


def _parse_one(
    sql: Union[str, exp.Expression],
    dialect: sqlglot.Dialect,
    schema: Optional[Union[dict, Schema]] = None,
):
    """Utility to make sure we parse/read queries with the correct dialect."""
    # https://www.sqlite.org/optoverview.html
    node = sql
    if isinstance(sql, str):
        node = parse_one(sql, dialect=dialect)
    if schema is not None:
        node = qualify(
            expression=node,
            dialect=dialect,
            schema=MappingSchema(schema, dialect=dialect, normalize=False),
            expand_alias_refs=False,
            expand_stars=False,
            quote_identifiers=False,
            allow_partial_qualification=True,
            validate_qualify_columns=False,
        )
    return node
