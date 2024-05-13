from typing import Generator, List, Dict, Collection
from typing import Iterable
import pandas as pd
from colorama import Fore
import logging
import re
from attr import attrib, attrs
from sqlalchemy.schema import CreateTable
from sqlalchemy import create_engine, inspect, MetaData
from sqlalchemy.sql import text
from sqlalchemy.engine import Engine, Connection
from pandas.io.sql import get_schema
from abc import abstractmethod

DOCS_TABLE_NAME = "documents"


@attrs(auto_detect=True)
class Database:
    db_path: str = attrib()
    db_prefix: str = attrib()

    engine: Engine = attrib(init=False)
    con: Connection = attrib(init=False)
    all_tables: List[str] = attrib(init=False)
    tablename_to_columns: Dict[str, Iterable] = attrib(init=False)

    def __attrs_post_init__(self):
        self.engine = create_engine(f"{self.db_prefix}{self.db_path}")
        self.con = self.engine.connect()
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)

    def _reset_connection(self):
        """Reset connection, so that temp tables are cleared."""
        self.con.close()
        self.con = self.engine.connect()

    @abstractmethod
    def has_temp_table(self, tablename: str) -> bool:
        """Temp tables are stored in different locations, depending on
        the DBMS. For example, sqlite puts them in `sqlite_temp_master`,
        and postgres goes in the main `information_schema.tables` with a
        'pg_temp' prefix.
        """
        ...

    @abstractmethod
    def get_sqlglot_schema(self) -> dict:
        """Returns database schema as a dictionary, in the format that
        sqlglot.optimizer expects.

        Examples:
            ```python
            db.get_sqlglot_schema()
            > {"x": {"A": "INT", "B": "INT", "C": "INT", "D": "INT", "Z": "STRING"}}
            ```
        """
        ...

    def tables(self):
        return inspect(self.engine).get_table_names()

    def iter_columns(self, tablename: str) -> Generator[str, None, None]:
        if tablename in self.tables():
            for column_data in inspect(self.engine).get_columns(tablename):
                yield column_data["name"]

    def schema_string(self, use_tables: Collection[str] = None) -> str:
        create_table_stmts = []
        for table in self.metadata.sorted_tables:
            if use_tables:
                if table.name not in use_tables:
                    continue
            create_table_stmts.append(str(CreateTable(table)).strip())
        return "\n\n".join(create_table_stmts)

    def to_temp_table(self, df: pd.DataFrame, tablename: str):
        if self.has_temp_table(tablename):
            self.con.execute(text(f'DROP TABLE "{tablename}"'))
        create_table_stmt = get_schema(df, name=tablename, con=self.con).strip()
        # Insert 'TEMP' keyword
        create_table_stmt = re.sub(
            r"^CREATE TABLE", "CREATE TEMP TABLE", create_table_stmt
        )
        logging.debug(Fore.CYAN + create_table_stmt + Fore.RESET)
        self.con.execute(text(create_table_stmt))
        df.to_sql(name=tablename, con=self.con, if_exists="append", index=False)

    def execute_query(self, query: str, params: dict = None) -> pd.DataFrame:
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
        return pd.read_sql(text(query), self.con, params=params)
