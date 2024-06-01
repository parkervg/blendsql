from typing import Dict, Optional, List, Collection, Type, Generator, Set, Union
import duckdb
from duckdb import DuckDBPyConnection
import pandas as pd
from colorama import Fore
from attr import attrs, attrib

from ._database import Database
from .._logger import logger


@attrs
class Pandas(Database):
    """A SQLite database connection.
    Can be initialized via a path to the database file.

    Examples:
        ```python
        from blendsql.db import SQLite
        db = SQLite("./path/to/database.db")
        ```
    """

    # Can be either a dict from name -> pd.DataFrame
    # or, a single pd.DataFrame object
    data: Union[Dict[str, pd.DataFrame], pd.DataFrame] = attrib()
    default_tablename: str = attrib(default="w")

    con: DuckDBPyConnection = attrib(init=False)
    # We use below to track which tables we should drop on '_reset_connection'
    temp_tables: Set[str] = set()

    def __attrs_post_init__(self):
        self.con = duckdb.connect(database=":memory:")
        if isinstance(self.data, pd.DataFrame):
            self.db_url = "Local pandas tables"
            # I don't really understand the scope of duckdb's replacement scan here
            # I assign the underlying data to _df, since passing self.data doesn't work
            # in the self.con.sql call.
            _df = self.data
            self.con.sql(f"CREATE TABLE {self.default_tablename} AS SELECT * FROM _df")

        elif isinstance(self.data, dict):
            self.db_url = f"Local pandas tables {', '.join(self.data.keys())}"
            for tablename, _df in self.data.items():
                # Note: duckdb.sql connects to the default in-memory database connection
                self.con.sql(f"CREATE TABLE {tablename} AS SELECT * FROM _df")
        else:
            raise ValueError(
                "Unknown datatype passed to `Pandas`!\nWe expect either a single dataframe, or a dictionary mapping many tables from {tablename: df}"
            )

    def _reset_connection(self):
        """Reset connection, so that temp tables are cleared."""
        for tablename in self.temp_tables:
            self.con.execute(f'DROP TABLE "{tablename}"')
        self.temp_tables = set()

    def tables(self) -> List[str]:
        return self.execute_to_list("SHOW TABLES;")

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

    def get_sqlglot_schema(self) -> dict:
        """Returns database schema as a dictionary, in the format that
        sqlglot.optimizer expects.

        Examples:
            ```python
            db.get_sqlglot_schema()
            > {"x": {"A": "INT", "B": "INT", "C": "INT", "D": "INT", "Z": "STRING"}}
            ```
        """
        # TODO
        return None

    def schema_string(self, use_tables: Collection[str] = None) -> str:
        """Converts the database to a series of 'CREATE TABLE' statements."""
        # TODO
        return None
