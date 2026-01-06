import sqlglot.dialects
from sqlglot.dialects import SQLite, Postgres
from sqlglot.schema import MappingSchema
from sqlglot import parse_one
from sqlglot.optimizer.qualify_columns import qualify_columns
from sqlglot.schema import Schema

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

        def _parse_conjunction(self):
            """Override conjunction parsing to handle bare BlendSQL functions in boolean context"""
            left = self._parse_equality()

            # If we have a bare BlendSQLFunction and the next token is AND/OR,
            # wrap it in = TRUE
            if isinstance(left, BlendSQLFunction) and self._match_set(
                (TokenType.AND, TokenType.OR)
            ):
                self._retreat(self._index - 1)  # Go back to reprocess the AND/OR
                left = exp.EQ(this=left, expression=exp.true())

            # Continue with normal conjunction parsing
            while self._match_set(self.CONJUNCTION):
                connector = self._prev
                right = self._parse_equality()

                # If right side is a bare BlendSQLFunction, wrap it
                if isinstance(right, BlendSQLFunction):
                    right = exp.EQ(this=right, expression=exp.true())

                left = self.expression(
                    self.CONJUNCTION.get(connector.token_type),
                    this=left,
                    expression=right,
                    comments=connector.comments,
                )

            return left

        def _parse_where(self, skip_where_token: bool = False) -> exp.Where | None:
            """Override WHERE parsing to handle bare BlendSQL functions"""
            if not skip_where_token and not self._match(TokenType.WHERE):
                return None

            condition = self._parse_assignment()

            # If the entire WHERE condition is just a bare BlendSQLFunction, wrap it
            if isinstance(condition, BlendSQLFunction):
                condition = exp.EQ(this=condition, expression=exp.true())

            return self.expression(
                exp.Where, comments=self._prev_comments, this=condition
            )

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


def _parse_one(
    sql: str | exp.Expression,
    dialect: sqlglot.Dialect,
    schema: dict | Schema | None = None,
):
    """Utility to make sure we parse/read queries with the correct dialect."""
    # https://www.sqlite.org/optoverview.html
    node = sql
    if isinstance(sql, str):
        node = parse_one(sql, dialect=dialect)
    if schema is not None:
        node = qualify_columns(
            expression=node,
            schema=MappingSchema(schema, dialect=dialect, normalize=False),
            expand_alias_refs=True,
            expand_stars=False,
            allow_partial_qualification=True,
            infer_schema=True,
            dialect=dialect,
        )

        if dialect.__name__ == "BlendSQLDuckDB":
            # Otherwise,
            #  ```
            #  SELECT content[0:5000] AS "README"
            #  FROM read_text('https://raw.githubusercontent.com/parkervg/blendsql/main/README.md')
            #  ``` becomes `"".content[0:5000]`
            if "read_" in sql.lower():

                def remove_empty_quoted_identifiers(node):
                    if (
                        isinstance(node, exp.Identifier)
                        and node.this == ""
                        and node.args.get("quoted") == True
                    ):
                        return None  # Remove this node
                    return node

                node = node.transform(remove_empty_quoted_identifiers)

    return node
