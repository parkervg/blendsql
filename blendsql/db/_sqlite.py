from pathlib import Path

from .utils import double_quote_escape
from ._database import Database


class SQLite(Database):
    def __init__(self, db_path: str):
        super().__init__(db_path=Path(db_path).resolve(), db_prefix="sqlite:///")

    def get_sqlglot_schema(self) -> dict:
        """Returns database schema as a dictionary, in the format that
        sqlglot.optimizer expects.

        Examples:
            >>> db.get_sqlglot_schema()
            {"x": {"A": "INT", "B": "INT", "C": "INT", "D": "INT", "Z": "STRING"}}
        """
        schema = {}
        for tablename in self.tables():
            schema[f'"{double_quote_escape(tablename)}"'] = {}
            for _, row in self.execute_query(
                f"""
            SELECT name, type FROM pragma_table_info(:t)
            """,
                {"t": tablename},
            ).iterrows():
                schema[f'"{double_quote_escape(tablename)}"'][
                    '"' + row["name"] + '"'
                ] = row["type"]
        return schema
