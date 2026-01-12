import importlib.util
from typing import Callable, Generator
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from functools import cached_property
import polars as pl
import pandas as pd

from blendsql.db.database import Database
from blendsql.db.utils import double_quote_escape
from blendsql.common.logger import logger, Color

_has_duckdb = importlib.util.find_spec("duckdb") is not None


@dataclass
class DuckDB(Database):
    """An in-memory DuckDB database connection.
    Can be initialized via any of the available class methods.

    Examples:
        ```python
        from blendsql.db import DuckDB
        db = DuckDB.from_pandas(
            pd.DataFrame(
                {
                    "name": ["John", "Parker"],
                    "age": [12, 26]
                },
            )
        )
        # Or, load multiple dataframes
        db = DuckDB.from_pandas(
            {
                "students": pd.DataFrame(
                    {
                        "name": ["John", "Parker"],
                        "age": [12, 26]
                    },
                ),
                "classes": pd.DataFrame(
                    {
                        "class": ["Physics 101", "Chemistry"],
                        "size": [50, 32]
                    },
                ),
            }
        )
        ```
    """

    # Can be either a dict from name -> pd.DataFrame
    # or, a single pd.DataFrame object
    con: "DuckDBPyConnection" = field()
    db_url: str = field(default=None)

    # We use below to track which tables we should drop on '_reset_connection'
    temp_tables: set[str] = field(default_factory=set)

    @classmethod
    def from_pandas(
        cls,
        data: dict[str, pd.DataFrame] | pd.DataFrame,
        tablename: str = "w",
    ):
        import duckdb

        con = duckdb.connect(database=":memory:")
        if isinstance(data, pd.DataFrame):
            db_url = "Local pandas table"
            # I don't really understand the scope of duckdb's replacement scan here
            # I assign the underlying data to _df, since passing self.data doesn't work
            # in the self.con.sql call.
            _df = data
            con.sql(
                f'CREATE TABLE "{double_quote_escape(tablename)}" AS SELECT * FROM _df'
            )

        elif isinstance(data, dict):
            db_url = f"Local pandas tables {', '.join(data.keys())}"
            for tablename, _df in data.items():
                # Note: duckdb.sql connects to the default in-memory database connection
                con.sql(
                    f'CREATE TABLE "{double_quote_escape(tablename)}" AS SELECT * FROM _df'
                )
        else:
            raise ValueError(
                "Unknown datatype passed to `Pandas`!\nWe expect either a single dataframe, or a dictionary mapping many tables from {tablename: df}"
            )
        return cls(con=con, db_url=db_url)

    @classmethod
    def from_sqlite(cls, filepath: str, additional_cmds: list[str] | None = None):
        import duckdb

        con = duckdb.connect(database=":memory:")
        db_url = str(Path(filepath).resolve())
        con.sql("INSTALL sqlite;")
        con.sql("LOAD sqlite;")
        if additional_cmds is not None:
            for cmd in additional_cmds:
                con.sql(cmd)
        con.sql(f"ATTACH '{db_url}' AS sqlite_db (TYPE sqlite);")
        con.sql("USE sqlite_db")
        return cls(con=con, db_url=db_url)

    @classmethod
    def from_file(cls, filepath: str):
        import duckdb

        con = duckdb.connect(filepath)
        db_url = str(Path(filepath).resolve())
        return cls(con=con, db_url=db_url)

    def _reset_connection(self):
        """Reset connection, so that temp tables are cleared."""
        for tablename in self.temp_tables:
            self.con.sql(f'DROP TABLE IF EXISTS "{tablename}"')
        self.temp_tables = set()

    def has_temp_table(self, tablename: str) -> bool:
        return tablename in self.execute_to_list("SHOW TABLES")

    @cached_property
    def sqlglot_schema(self) -> dict:
        """Returns database schema as a dictionary, in the format that
        sqlglot.optimizer expects.

        Examples:
            ```python
            db.sqlglot_schema
            > {"x": {"A": "INT", "B": "INT", "C": "INT", "D": "INT", "Z": "STRING"}}
            ```
        """
        schema: dict[str, dict] = {}
        for tablename in self.tables():
            schema[tablename] = {}
            for column_name, column_type in self.con.sql(
                f'SELECT column_name, column_type FROM (DESCRIBE "{double_quote_escape(tablename)}")'
            ).fetchall():
                schema[tablename][column_name] = column_type
        return schema

    def tables(self) -> list[str]:
        return self.execute_to_list("SHOW TABLES;")

    def iter_columns(self, tablename: str) -> Generator[str, None, None]:
        for row in self.con.sql(
            f'SELECT column_name FROM (DESCRIBE "{double_quote_escape(tablename)}")'
        ).fetchall():
            yield row[0]

    def schema_string(self, use_tables: Collection[str] | None = None) -> str:
        """Converts the database to a series of 'CREATE TABLE' statements."""
        # TODO
        return None

    def to_temp_table(self, df: pd.DataFrame, tablename: str):
        """Technically, when duckdb is run in-memory (as is the default),
        all created tables are temporary tables (since they expire at the
        end of the session). So, we don't really need to insert 'TEMP' keyword here?
        """
        # DuckDB has this cool 'CREATE OR REPLACE' syntax
        # https://duckdb.org/docs/sql/statements/create_table.html#create-or-replace
        create_table_stmt = (
            f'CREATE OR REPLACE TEMP TABLE "{tablename}" AS SELECT * FROM df'
        )
        logger.debug(Color.quiet_sql(create_table_stmt))
        self.con.sql(create_table_stmt)
        self.temp_tables.add(tablename)
        logger.debug(Color.update(f"Created temp table {tablename}"))

    def execute_to_df(
        self, query: str, lazy=True, close_conn=True, **_
    ) -> pl.LazyFrame:
        """On params with duckdb: https://github.com/duckdb/duckdb/issues/9853#issuecomment-1832732933

        If `close_conn==True` and `lazy=True`, we can't call `self.con.sql(query).pl(lazy=True)`,
            since this leaves an open connection that blocks future queries.
            Instead, we create a pl.DataFrame and call `.lazy()` on it.
        """
        if close_conn:
            res = self.con.sql(query).pl()
            return res.lazy() if lazy else res
        else:
            return self.con.sql(query).pl(lazy=lazy)

    def execute_to_list(
        self, query: str, to_type: Callable | None = lambda x: x
    ) -> list:
        result = self.con.sql(query).fetchall()
        return [to_type(row[0]) for row in result]
