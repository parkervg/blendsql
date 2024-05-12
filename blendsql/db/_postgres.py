from pathlib import Path
import importlib.util

from ._database import Database

_has_psycopg2 = importlib.util.find_spec("psycopg2") is not None


class PostreSQL(Database):
    def __init__(self, db_path: str):
        if not _has_psycopg2:
            raise ImportError(
                "Please install psycopg2 with `pip install psycopg2`!"
            ) from None
        super().__init__(db_path=Path(db_path).resolve(), db_prefix="postgresql:///")
