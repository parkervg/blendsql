from typing import Generator, List, Callable, Optional, Union
from collections.abc import Collection
import pandas as pd
from colorama import Fore
import re
from attr import attrib, attrs
from sqlalchemy.schema import CreateTable
from sqlalchemy import create_engine, inspect, MetaData
from sqlalchemy.sql import text
from sqlalchemy.engine import Engine, Connection, URL
from pandas.io.sql import get_schema

from ._database import Database
from .._logger import logger
from .utils import double_quote_escape, truncate_df_content, LazyTables
from .bridge_content_encoder import get_database_matches


@attrs(auto_detect=True)
class SQLAlchemyDatabase(Database):
    db_url: URL = attrib()

    engine: Engine = attrib(init=False)
    con: Connection = attrib(init=False)

    def __attrs_post_init__(self):
        self.lazy_tables = LazyTables()
        self.engine = create_engine(self.db_url)
        self.con = self.engine.connect()
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)

    def _reset_connection(self):
        """Reset connection, so that temp tables are cleared."""
        self.con.close()
        self.con = self.engine.connect()

    def tables(self) -> List[str]:
        return inspect(self.engine).get_table_names()

    def iter_columns(self, tablename: str) -> Generator[str, None, None]:
        if tablename in self.tables():
            for column_data in inspect(self.engine).get_columns(tablename):
                yield column_data["name"]

    def schema_string(self, use_tables: Optional[Collection[str]] = None) -> str:
        create_table_stmts = []
        for table in self.metadata.sorted_tables:
            if use_tables:
                if table.name not in use_tables:
                    continue
            create_table_stmts.append(str(CreateTable(table)).strip())
        return "\n\n".join(create_table_stmts)

    def to_serialized(
        self,
        num_rows: int = 3,
        truncate_content: int = 300,
        use_tables: Optional[Collection[str]] = None,
        include_content: Union[str, Collection[str]] = "all",
        use_bridge_encoder: bool = False,
        question: Optional[str] = None,
    ) -> str:
        """Returns a string representation of the database, with example rows."""
        serialized_db = []
        for tablename in self.tables():
            if use_tables is not None:
                if tablename not in use_tables:
                    continue
            serialized_db.append(self.schema_string(use_tables=[tablename]))
            if include_content != "all":
                if tablename not in include_content:
                    serialized_db.append("\n")
                    continue
            get_rows_query = (
                f'SELECT * FROM "{double_quote_escape(tablename)}" LIMIT {num_rows}'
            )
            total_num_rows = self.execute_to_list(
                f'SELECT COUNT(*) FROM "{double_quote_escape(tablename)}"'
            )[0]
            serialized_db.append("/*")
            serialized_db.append(
                f"{num_rows} example rows:"
                if num_rows < total_num_rows
                else "Entire table:"
            )
            rows = self.execute_to_df(get_rows_query)
            if truncate_content is not None:
                # Truncate long strings
                rows = truncate_df_content(rows, truncate_content)
                serialized_db.append(get_rows_query)
            serialized_db.append(f"{rows.to_string(index=False)}")
            serialized_db.append("*/\n")
            if use_bridge_encoder:
                bridge_hints = []
                column_str_with_values = "{table}.{column} ( {values} )"
                value_sep = " , "
                for columnname in self.iter_columns(tablename):
                    matches = get_database_matches(
                        question=question,
                        table_name=tablename,
                        column_name=columnname,
                        db=self,
                    )
                    if matches:
                        bridge_hints.append(
                            column_str_with_values.format(
                                table=tablename,
                                column=columnname,
                                values=value_sep.join(matches),
                            )
                        )
                if len(bridge_hints) > 0:
                    serialized_db.append(
                        "Here are some values that may be useful: "
                        + " , ".join(bridge_hints)
                    )
        return "\n".join(serialized_db).strip()

    def to_temp_table(self, df: pd.DataFrame, tablename: str):
        if self.has_temp_table(tablename):
            self.con.execute(text(f'DROP TABLE "{tablename}"'))
        create_table_stmt = get_schema(df, name=tablename, con=self.con).strip()
        # Insert 'TEMP' keyword
        create_table_stmt = re.sub(
            r"^CREATE TABLE", "CREATE TEMP TABLE", create_table_stmt
        )
        logger.debug(Fore.CYAN + create_table_stmt + Fore.RESET)
        self.con.execute(text(create_table_stmt))
        df.to_sql(name=tablename, con=self.con, if_exists="append", index=False)

    def execute_to_df(self, query: str, params: Optional[dict] = None) -> pd.DataFrame:
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

    def execute_to_list(self, query: str, to_type: Callable = lambda x: x) -> list:
        """A lower-level execute method that doesn't use the pandas processing logic.
        Returns results as a tuple.
        """
        res = []
        for row in self.con.execute(text(query)).fetchall():
            res.append(to_type(row[0]))
        return res
