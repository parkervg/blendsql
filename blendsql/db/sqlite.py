import sqlite3
from pathlib import Path
from sqlite3 import OperationalError
from typing import Generator, Tuple, Union, List, Dict
from typing import Iterable
import logging
from functools import lru_cache
import pandas as pd
from attr import attrib, attrs
from colorama import Fore

from .utils import single_quote_escape, double_quote_escape


@attrs(auto_detect=True)
class SQLite:
    """
    Class used to connect to a SQLite database.

    Args:
        db_path: Path to the db
    """

    db_path: str = attrib()
    con: sqlite3.Connection = attrib(init=False)

    all_tables: List[str] = attrib(init=False)
    tablename_to_columns: Dict[str, Iterable] = attrib(init=False)

    def __attrs_post_init__(self):
        if not Path(self.db_path).exists():
            raise ValueError(f"{self.db_path} does not exist")
        try:
            self.con = sqlite3.connect(self.db_path)
        except OperationalError:
            print(self.db_path)
            raise
        self.all_tables = None
        self.tablename_to_columns = {}

    def has_table(self, tablename: str) -> bool:
        if (
            self.execute_query(
                f"SELECT count(*) FROM sqlite_master WHERE type='table' AND name='{single_quote_escape(tablename)}';"
            ).values.item()
            > 0
        ):
            return True
        return False

    def iter_tables(
        self, use_tables: Iterable[str] = None
    ) -> Generator[str, None, None]:
        for tablename in self._iter_tables():
            if use_tables is not None and tablename not in use_tables:
                continue
            yield tablename

    def iter_columns(self, tablename: str) -> Generator[str, None, None]:
        for columname in self._get_columns(tablename=tablename):
            yield columname

    def create_clauses(self) -> Generator[Tuple[str, str], None, None]:
        for tablename in self._iter_tables():
            create_clause = _create_clause(con=self.con, tablename=tablename)
            yield (tablename, create_clause)

    def create_clause(self, tablename):
        return (tablename, _create_clause(con=self.con, tablename=tablename))

    def get_sqlglot_schema(self) -> dict:
        """Returns database schema as a dictionary, in the format that
        sqlglot.optimizer expects.

        Examples:
            >>> db.get_sqlglot_schema()
            {"x": {"A": "INT", "B": "INT", "C": "INT", "D": "INT", "Z": "STRING"}}
        """
        schema = {}
        for tablename in self._iter_tables():
            schema[tablename] = {}
            for _, row in self.execute_query(
                f"""
            SELECT name, type FROM pragma_table_info('{tablename}')
            """
            ).iterrows():
                schema[tablename]['"' + row["name"] + '"'] = row["type"]
        return schema

    def to_serialized(
        self,
        ignore_tables: Iterable[str] = None,
        num_rows: int = 0,
        table_description: str = None,
    ) -> str:
        """Generates a string representation of a database, via `CREATE` statements.
        This can then be passed to a LLM as context.

        Args:
            ignore_tables: Name of tables to ignore in serialization. Default is just 'documents'.
            num_rows: How many rows per table to include in serialization
            table_description: Optional table description to add at top
        """
        if ignore_tables is None:
            ignore_tables = {"documents"}
        serialized_db = (
            []
            if table_description is None
            else [f"Table Description: {table_description}\n"]
        )
        for tablename, create_clause in self.create_clauses():
            if tablename in ignore_tables:
                continue
            serialized_db.append(create_clause)
            serialized_db.append("\n")
            if num_rows > 0:
                get_rows_query = (
                    f'SELECT * FROM "{double_quote_escape(tablename)}" LIMIT {num_rows}'
                )
                serialized_db.append("\n/*")
                serialized_db.append(f"\n{num_rows} example rows:")
                serialized_db.append(f"\n{get_rows_query}")
            else:
                continue
            rows = self.execute_query(get_rows_query)
            # Truncate long strings
            rows = rows.apply(
                lambda x: f"{str(x)[:50]}..."
                if isinstance(x, str) and len(str(x)) > 50
                else x,
                axis=1,
            )
            serialized_db.append(f"\n{rows.to_string(index=False)}")
            serialized_db.append("\n*/")
        serialized_db = "\n\n".join(serialized_db).strip()
        return serialized_db

    # @profile
    def execute_query(
        self, query: str, return_error: bool = False, silence_errors: bool = True
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
        """
        Execute the given query.
        """
        try:
            df = pd.read_sql(query, self.con)
            if return_error:
                return (df, None)
            return df
        except Exception as e:
            if silence_errors:
                print(Fore.RED + "Something went wrong!" + Fore.RESET)
                print(e)
                if return_error:
                    return (pd.DataFrame(), str(e))
                return pd.DataFrame()  # Return empty pd.DataFrame
            raise (e)

    def _iter_tables(self):
        if self.all_tables is None:
            self.all_tables = pd.read_sql(
                "SELECT tbl_name FROM sqlite_master WHERE type='table'", self.con
            )["tbl_name"]
        return self.all_tables

    def _get_columns(self, tablename: str):
        if tablename not in self.tablename_to_columns:
            self.tablename_to_columns[tablename] = pd.read_sql(
                f'PRAGMA table_info("{double_quote_escape(tablename)}");', self.con
            )["name"]
        return self.tablename_to_columns[tablename]


@lru_cache(maxsize=1000, typed=False)
def _create_clause(con, tablename) -> str:
    create_clause = pd.read_sql(
        f'SELECT sql FROM sqlite_master WHERE tbl_name = "{double_quote_escape(tablename)}"',
        con,
    )
    if create_clause.size < 1:
        logging.debug(
            f"Expected create_clause size to be 1, got {create_clause.size}\n{create_clause}"
        )
        return ""
    return create_clause.values[0][0]
