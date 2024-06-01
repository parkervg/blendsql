---
hide:
  - toc
---
# Databases

Since BlendSQL relies on the package [sqlglot](https://github.com/tobymao/sqlglot) for query optimization (which supports a [wide variety of SQL dialects](https://github.com/tobymao/sqlglot/blob/main/sqlglot/dialects/__init__.py)) and the notion of [temporary tables](https://en.wikibooks.org/wiki/Structured_Query_Language/Temporary_Table), it can easily integrate with many different SQL DBMS. 

Currently, the following are supported.

- [DuckDB](./duckdb.md)
  - Allows for reading local data structures like pandas DataFrames
- [SQLite](./sqlite.md)
- [PostgreSQL](./postgresql.md)

::: blendsql.db._database.Database
    handler: python
    show_source: true