import typing as t
import pandas as pd
from attr import attrib
from sqlalchemy.engine import URL
from abc import abstractmethod, ABC

from blendsql.db.utils import LazyTables


class Database(ABC):
    db_url: t.Union[URL, str] = attrib()
    lazy_tables: LazyTables = LazyTables()

    def __str__(self):
        return f"{self.__class__} @ {self.db_url}"

    def __repr__(self):
        return f"{self.__class__} @ {self.db_url}"

    @abstractmethod
    def _reset_connection(self) -> None:
        """Reset connection, so that temp tables are cleared."""
        ...

    @abstractmethod
    def has_temp_table(self, tablename: str) -> bool:
        """Temp tables are stored in different locations, depending on
        the DBMS. For example, sqlite puts them in `sqlite_temp_master`,
        and postgres goes in the main `information_schema.tables` with a
        'pg_temp' prefix.
        """
        ...

    @property
    @abstractmethod
    def sqlglot_schema(self) -> dict:
        """Returns database schema as a dictionary, in the format that
        sqlglot.optimizer expects.

        Examples:
            ```python
            db.sqlglot_schema
            > {"x": {"A": "INT", "B": "INT", "C": "INT", "D": "INT", "Z": "STRING"}}
            ```
        """
        ...

    @abstractmethod
    def tables(self) -> t.List[str]:
        """Get all table names associated with a database."""
        ...

    @abstractmethod
    def iter_columns(self, tablename: str) -> t.Generator[str, None, None]:
        """Yield all column names associated with a tablename."""
        ...

    @abstractmethod
    def schema_string(self, use_tables: t.Optional[t.Collection[str]] = None) -> str:
        """Converts the database to a series of 'CREATE TABLE' statements."""

    @abstractmethod
    def to_temp_table(self, df: pd.DataFrame, tablename: str):
        """Write the given pandas dataframe as a temp table 'tablename'."""
        ...

    @abstractmethod
    def execute_to_df(
        self, query: str, params: t.Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Execute the given query and return results as dataframe.

        Args:
            query: The SQL query to execute. Can use `named` paramstyle from PEP 249
                https://peps.python.org/pep-0249/#paramstyle
            params: Dict containing mapping from name to value.

        Returns:
            pd.DataFrame

        Examples:
            ```python
            from blendsql.db import SQLite
            db = SQLite("./path/to/database.db")
            db.execute_query("SELECT * FROM t WHERE c = :v", {"v": "value"})
            ```
        """
        ...

    @abstractmethod
    def execute_to_list(self, query: str, to_type: t.Callable = lambda x: x) -> list:
        """A lower-level execute method that doesn't use the pandas processing logic.
        Returns results as a list.
        """
        ...
