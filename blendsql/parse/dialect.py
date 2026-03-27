import re
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

BLENDSQL_FUNC_PREFIX = "__BSQL__"


def is_blendsql_node(node) -> bool:
    return (
        isinstance(node, exp.Anonymous)
        and isinstance(node.this, str)
        and node.this.startswith(BLENDSQL_FUNC_PREFIX)
    )


def get_blendsql_func_name(node) -> str:
    return node.this[len(BLENDSQL_FUNC_PREFIX) :]


def get_blendsql_fn_args(node) -> list:
    return [e for e in (node.expressions or []) if not isinstance(e, exp.EQ)]


def get_blendsql_fn_kwargs(node) -> list:
    return [e for e in (node.expressions or []) if isinstance(e, exp.EQ)]


class _BlendSQLFunctionMeta(type):
    def __instancecheck__(cls, instance):
        return is_blendsql_node(instance)


class BlendSQLFunction(metaclass=_BlendSQLFunctionMeta):
    """Sentinel class for isinstance checks on BlendSQL function AST nodes.

    BlendSQL function nodes are represented as exp.Anonymous instances with
    a BLENDSQL_FUNC_PREFIX prefix. Use BlendSQLFunction(this=name) to create
    one (returns exp.Anonymous), and isinstance(node, BlendSQLFunction) to check.
    """

    def __new__(cls, this, fn_args=None, fn_kwargs=None, **kwargs):
        return exp.Anonymous(
            this=BLENDSQL_FUNC_PREFIX + this,
            expressions=(fn_args or []) + (fn_kwargs or []),
        )


exp.BlendSQLFunction = BlendSQLFunction


def _preprocess_blendsql_syntax(sql: str) -> str:
    """Replace {{ FunctionName(...) }} with BLENDSQL_FUNC_PREFIX+FunctionName(...).
    Handles nested {{ }} blocks via depth counting."""
    result = []
    i = 0
    n = len(sql)
    in_single_quote = False
    in_double_quote = False

    while i < n:
        c = sql[i]

        if in_single_quote:
            result.append(c)
            if c == "'" and (i == 0 or sql[i - 1] != "\\"):
                in_single_quote = False
            i += 1
            continue

        if in_double_quote:
            result.append(c)
            if c == '"' and (i == 0 or sql[i - 1] != "\\"):
                in_double_quote = False
            i += 1
            continue

        if c == "'":
            in_single_quote = True
            result.append(c)
            i += 1
            continue

        if c == '"':
            in_double_quote = True
            result.append(c)
            i += 1
            continue

        if sql[i : i + 2] == "{{":
            # Find matching "}}" using depth counting, respecting nested {{ }} and quotes
            depth = 1
            j = i + 2
            inner_in_single = False
            inner_in_double = False
            while j < n and depth > 0:
                d = sql[j]
                if inner_in_single:
                    if d == "'" and sql[j - 1] != "\\":
                        inner_in_single = False
                elif inner_in_double:
                    if d == '"' and sql[j - 1] != "\\":
                        inner_in_double = False
                elif d == "'":
                    inner_in_single = True
                elif d == '"':
                    inner_in_double = True
                elif sql[j : j + 2] == "{{":
                    depth += 1
                    j += 1  # skip second {
                elif sql[j : j + 2] == "}}":
                    depth -= 1
                    if depth == 0:
                        break
                    j += 1  # skip second }
                j += 1
            if depth == 0:
                content = sql[i + 2 : j].strip()
                # Recursively process nested {{ }} in the content
                content = _preprocess_blendsql_syntax(content)
                result.append(BLENDSQL_FUNC_PREFIX + content)
                i = j + 2  # skip past "}}"
            else:
                result.append(c)
                i += 1
        else:
            result.append(c)
            i += 1

    return "".join(result)


def _postprocess_blendsql_sql(sql: str) -> str:
    """Convert __BSQL__FuncName(...) back to {{FuncName(...)}}."""
    if BLENDSQL_FUNC_PREFIX not in sql:
        return sql
    result = []
    i = 0
    n = len(sql)
    prefix_len = len(BLENDSQL_FUNC_PREFIX)

    while i < n:
        idx = sql.find(BLENDSQL_FUNC_PREFIX, i)
        if idx == -1:
            result.append(sql[i:])
            break
        result.append(sql[i:idx])
        j = idx + prefix_len
        while j < n and (sql[j].isalnum() or sql[j] == "_"):
            j += 1
        func_name = sql[idx + prefix_len : j]
        if j < n and sql[j] == "(":
            depth = 1
            k = j + 1
            in_single = False
            in_double = False
            while k < n and depth > 0:
                c = sql[k]
                if in_single:
                    if c == "'" and sql[k - 1] != "\\":
                        in_single = False
                elif in_double:
                    if c == '"' and sql[k - 1] != "\\":
                        in_double = False
                elif c == "'":
                    in_single = True
                elif c == '"':
                    in_double = True
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                k += 1
            inner = sql[j + 1 : k - 1]
            result.append(f"{{{{{func_name}({inner})}}}}")
            i = k
        else:
            result.append(BLENDSQL_FUNC_PREFIX + func_name)
            i = j

    return "".join(result)


def _wrap_bare_blendsql(node):
    """Post-parse transform: wrap bare BlendSQL function nodes in = TRUE."""
    if isinstance(node, (exp.And, exp.Or)):
        if is_blendsql_node(node.this):
            node.set("this", exp.EQ(this=node.this, expression=exp.true()))
        if is_blendsql_node(node.expression):
            node.set("expression", exp.EQ(this=node.expression, expression=exp.true()))
    elif isinstance(node, exp.Where):
        if is_blendsql_node(node.this):
            node.set("this", exp.EQ(this=node.this, expression=exp.true()))
    return node


class BlendSQLDialect(sqlglot.Dialect):
    class Tokenizer(sqlglot.Tokenizer):
        pass

    def generate(self, expression, **opts) -> str:
        sql = super().generate(expression, **opts)
        return _postprocess_blendsql_sql(sql)


class BlendSQLDuckDB(BlendSQLDialect, DuckDB):
    class Tokenizer(BlendSQLDialect.Tokenizer, DuckDB.Tokenizer):
        pass


class BlendSQLPostgres(BlendSQLDialect, Postgres):
    class Tokenizer(BlendSQLDialect.Tokenizer, Postgres.Tokenizer):
        pass


class BlendSQLSQLite(BlendSQLDialect, SQLite):
    class Tokenizer(BlendSQLDialect.Tokenizer, SQLite.Tokenizer):
        KEYWORDS = {
            **SQLite.Tokenizer.KEYWORDS,
            "MATCH": TokenType.GLOB,
        }

    def generate(self, expression, **opts) -> str:
        sql = super().generate(expression, **opts)
        # SQLite FTS5: reconvert GLOB back to MATCH
        # (The tokenizer maps MATCH -> GLOB token type for parsing)
        return re.sub(r"\bGLOB\b", "MATCH", sql)


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
    node = sql
    if isinstance(sql, str):
        modified_sql = _preprocess_blendsql_syntax(sql)
        node = parse_one(modified_sql, dialect=dialect)
        node = node.transform(_wrap_bare_blendsql)
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

        if isinstance(sql, str) and dialect.__name__ == "BlendSQLDuckDB":
            # Otherwise,
            #  ```
            #  SELECT content[0:5000] AS "README"
            #  FROM read_text('https://raw.githubusercontent.com/parkervg/blendsql/main/README.md')
            #  ``` becomes `"".content[0:5000]`
            if " read_" in sql.lower():

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
