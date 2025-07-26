import typing as t
import importlib.util
from sqlalchemy.engine import make_url, URL
from colorama import Fore
import logging
from functools import cached_property

from blendsql.db.sqlalchemy import SQLAlchemyDatabase

_has_psycopg2 = importlib.util.find_spec("psycopg2") is not None


class PostgreSQL(SQLAlchemyDatabase):
    """A PostgreSQL database connection.
    Can be initialized via the SQLAlchemy input string.
    https://docs.sqlalchemy.org/en/20/core/engines.html#postgresql

    Examples:
        ```python
        from blendsql.db import PostgreSQL
        db = PostgreSQL("user:password@localhost/mydatabase")
        ```
    """

    def __init__(self, db_path: str):
        if not _has_psycopg2:
            raise ImportError(
                "Please install psycopg2 with `pip install psycopg2-binary`!"
            ) from None
        db_url: URL = make_url(f"postgresql+psycopg2://{db_path}")
        if db_url.username is None:
            logging.warning(
                Fore.RED
                + "Connecting to postgreSQL database without specifying user!\nIt is strongly encouraged to create a `blendsql` user with read-only permissions and temp table creation privileges."
            )
        super().__init__(db_url=db_url)

    def has_temp_table(self, tablename: str) -> bool:
        return tablename in self.execute_to_list(
            "SELECT table_name FROM information_schema.tables WHERE table_schema LIKE 'pg_temp_%'"
        )

    @cached_property
    def sqlglot_schema(self) -> dict:
        schema: t.Dict[str, dict] = {}
        for tablename in self.tables():
            schema[tablename] = {}
            for _, row in self.execute_to_df(
                """
                    SELECT column_name as name, data_type as type 
                    FROM information_schema.columns 
                    WHERE table_name = :t
                    AND table_schema = 'public'
                    """,
                {"t": tablename},
            ).iterrows():
                schema[tablename][row["name"]] = row["type"]
        return schema
