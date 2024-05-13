import importlib.util

from ._database import Database

_has_psycopg2 = importlib.util.find_spec("psycopg2") is not None


class PostgreSQL(Database):
    """A Post"""

    def __init__(self, db_path: str):
        if not _has_psycopg2:
            raise ImportError(
                "Please install psycopg2 with `pip install psycopg2`!"
            ) from None
        super().__init__(db_path=db_path, db_prefix="postgresql+psycopg2://")

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
