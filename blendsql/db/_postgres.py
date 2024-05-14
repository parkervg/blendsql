import importlib.util
from sqlalchemy.engine import make_url, URL
from colorama import Fore
import logging

from ._database import Database

_has_psycopg2 = importlib.util.find_spec("psycopg2") is not None


class PostgreSQL(Database):
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
        return (
            tablename
            in self.execute_query(
                "SELECT * FROM information_schema.tables WHERE table_schema LIKE 'pg_temp_%'"
            )["table_name"].unique()
        )

    def get_sqlglot_schema(self) -> dict:
        # TODO
        return None
