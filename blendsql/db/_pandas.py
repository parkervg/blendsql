from typing import Dict, Optional, Type, Generator, Set
import duckdb
from duckdb import DuckDBPyConnection
import pandas as pd
from colorama import Fore
from attr import attrs, attrib

from .utils import LazyTables
from .._logger import logger


@attrs
class Pandas:
    """A SQLite database connection.
    Can be initialized via a path to the database file.

    Examples:
        ```python
        from blendsql.db import SQLite
        db = SQLite("./path/to/database.db")
        ```
    """

    name_to_df: Dict[str, pd.DataFrame] = attrib()

    con: DuckDBPyConnection = attrib(init=False)
    db_url: str = attrib(init=False)
    temp_tables: Set[str] = set()

    def __attrs_post_init__(self):
        self.con = duckdb.connect(database=":memory:")
        self.db_url = f"Local pandas tables {', '.join(self.name_to_df.keys())}"
        for tablename, _df in self.name_to_df.items():
            # Note: duckdb.sql connects to the default in-memory database connection
            self.con.sql(f"CREATE TABLE {tablename} AS SELECT * FROM df")
        self.lazy_tables = LazyTables()
        # We use below to track which tables we should drop on '_reset_connection'
        self.temp_tables = set()

    def _reset_connection(self):
        """Reset connection, so that temp tables are cleared."""
        for tablename in self.temp_tables:
            self.con.execute(f'DROP TABLE "{tablename}"')
        self.temp_tables = set()

    def to_temp_table(self, df: pd.DataFrame, tablename: str):
        """Technically, when duckdb is run in-memory (as is the default),
        all created tables are temporary tables (since they expire at the
        end of the session). So, we don't need to insert 'TEMP' keyword here.
        """
        self.con.sql(f'CREATE TABLE "{tablename}" AS SELECT * FROM df')
        self.temp_tables.add(tablename)
        logger.debug(Fore.CYAN + f"Created temp table {tablename}" + Fore.RESET)

    def has_temp_table(self, tablename: str) -> bool:
        return tablename in self.execute_to_list("SHOW TABLES")

    def iter_columns(self, tablename: str) -> Generator[str, None, None]:
        for row in self.con.execute(
            f"SELECT column_name FROM (DESCRIBE {tablename})"
        ).fetchall():
            yield row[0]

    def execute_to_df(self, query: str, params: dict = None) -> pd.DataFrame:
        """On params with duckdb: https://github.com/duckdb/duckdb/issues/9853#issuecomment-1832732933"""
        return self.con.sql(query).df()

    def execute_to_list(
        self, query: str, to_type: Optional[Type] = lambda x: x
    ) -> list:
        res = []
        for row in self.con.execute(query).fetchall():
            res.append(to_type(row[0]))
        return res
