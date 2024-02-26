from sqlglot.dialects import SQLite
from sqlglot.tokens import TokenType
from sqlglot.optimizer.qualify_columns import qualify_columns
from sqlglot.schema import Schema
from sqlglot import exp, parse_one
from typing import Union, Optional


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
        node = qualify_columns(expression=node, schema=schema)
    return node
