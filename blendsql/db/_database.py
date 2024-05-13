from typing import Generator, List, Dict, Collection
from typing import Iterable
import pandas as pd
from colorama import Fore
import logging
from attr import attrib, attrs
from sqlalchemy.schema import CreateTable
from sqlalchemy import create_engine, inspect, MetaData
from sqlalchemy.sql import text
from sqlalchemy.engine import Engine, Connection


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
        self.inspector = inspect(self.engine)
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)

    def has_table(self, tablename: str) -> bool:
        return self.inspector.has_table(tablename)

    def tables(self):
        return inspect(self.engine).get_table_names()

    def iter_columns(self, tablename: str) -> Generator[str, None, None]:
        for column_data in self.inspector.get_columns(tablename):
            yield column_data["name"]

    def drop_table(self, tablename):
        table = self.metadata.tables.get(tablename, None)
        if table is None:
            logging.debug(Fore.RED + f"No table found {tablename}" + Fore.RESET)
            return
        table.drop()

    def schema_string(self, use_tables: Collection[str] = None) -> str:
        create_table_stmts = []
        for table in self.metadata.sorted_tables:
            if use_tables:
                if table.name not in use_tables:
                    continue
            create_table_stmts.append(str(CreateTable(table)).strip())
        return "\n\n".join(create_table_stmts)

    def execute_query(self, query: str, params: dict = None) -> pd.DataFrame:
        """
        Execute the given query.

        Args:
            query: The SQL query to execute. Can use `named` paramstyle from PEP 249
                https://peps.python.org/pep-0249/#paramstyle
            params: Dict containing mapping from name to value.

        Returns:
            pd.DataFrame

        Example:
            >>> execute_query("SELECT * FROM t WHERE c = :v", {"v": "value"})
        """
        return pd.read_sql(text(query), self.con, params=params)
