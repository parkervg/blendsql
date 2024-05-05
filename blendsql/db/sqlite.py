import sqlite3
from pathlib import Path
from sqlite3 import OperationalError
from typing import Generator, Tuple, List, Dict, Set, Optional
from typing import Iterable
import logging
from functools import lru_cache
import pandas as pd
from attr import attrib, attrs
import re

from .utils import single_quote_escape, double_quote_escape, truncate_df_content


DOCS_TABLE_NAME = "documents"


@attrs(auto_detect=True)
class SQLite:
    """
    Class used to connect to a SQLite database.

    Args:
        db_path: Path to the db
    """

    db_path: str = attrib()
    check_same_thread: bool = attrib(default=True)

    con: sqlite3.Connection = attrib(init=False)
    all_tables: List[str] = attrib(init=False)
    tablename_to_columns: Dict[str, Iterable] = attrib(init=False)

    def __attrs_post_init__(self):
        if not Path(self.db_path).exists():
            raise ValueError(f"{self.db_path} does not exist")
        try:
            self.con = sqlite3.connect(
                self.db_path, check_same_thread=self.check_same_thread
            )
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
        self, use_tables: Optional[Iterable[str]] = None
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
            schema[f'"{double_quote_escape(tablename)}"'] = {}
            for _, row in self.execute_query(
                f"""
            SELECT name, type FROM pragma_table_info("{double_quote_escape(tablename)}")
            """
            ).iterrows():
                schema[f'"{double_quote_escape(tablename)}"'][
                    '"' + row["name"] + '"'
                ] = row["type"]
        return schema

    def to_serialized(
        self,
        ignore_tables: Optional[Iterable[str]] = None,
        use_tables: Optional[Set[str]] = None,
        num_rows: Optional[int] = 3,
        tablename_to_description: Optional[dict] = None,
        whole_table: Optional[bool] = False,
        truncate_content: Optional[int] = None,
        truncate_content_tokens: Optional[int] = None,
        tokenizer=None,
    ) -> str:
        """Generates a string representation of a database, via `CREATE` statements.
        This can then be passed to a LLM as context.

        Args:
            ignore_tables: Name of tables to ignore in serialization. Default is just 'documents'.
            num_rows: How many rows per table to include in serialization
            table_description: Optional table description to add at top
        """
        if all(x is not None for x in [ignore_tables, use_tables]):
            raise ValueError("Both `ignore_tables` and `use_tables` cannot be passed!")
        if ignore_tables is None:
            ignore_tables = set()
        serialized_db = []
        if use_tables:
            _create_clause_iter = [
                self.create_clause(tablename) for tablename in use_tables
            ]
        else:
            _create_clause_iter = self.create_clauses()
        for tablename, create_clause in _create_clause_iter:
            # Check if it's an artifact of virtual table, and ignore
            if re.search(r"^{}_".format(DOCS_TABLE_NAME), tablename):
                continue
            if tablename in ignore_tables:
                continue
            if use_tables is not None and tablename not in use_tables:
                continue
            if tablename_to_description is not None:
                if tablename in tablename_to_description:
                    if tablename_to_description[tablename] is not None:
                        serialized_db.append(
                            f"Table Description: {tablename_to_description[tablename]}"
                        )
            if not whole_table:
                serialized_db.append(f"{create_clause}")
            if (
                num_rows > 0 and not tablename.startswith(DOCS_TABLE_NAME)
            ) or whole_table:
                get_rows_query = (
                    f'SELECT * FROM "{double_quote_escape(tablename)}" LIMIT {num_rows}'
                    if not whole_table
                    else f'SELECT * FROM "{double_quote_escape(tablename)}"'
                )
                serialized_db.append("/*")
                if whole_table:
                    serialized_db.append("Entire table:")
                else:
                    serialized_db.append(f"{num_rows} example rows:")
                serialized_db.append(f"{get_rows_query}")
                rows = self.execute_query(get_rows_query)
                if truncate_content is not None:
                    rows = truncate_df_content(rows, truncate_content)
                serialized_db.append(f"{rows.to_string(index=False)}")
                serialized_db.append("*/\n")
        serialized_db = "\n".join(serialized_db).strip()
        return serialized_db

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute the given query.
        """
        return pd.read_sql(query, self.con)

    def _iter_tables(self):
        if self.all_tables is None:
            self.all_tables = pd.read_sql(
                "SELECT tbl_name FROM sqlite_master WHERE type='table'", self.con
            )["tbl_name"].tolist()
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
