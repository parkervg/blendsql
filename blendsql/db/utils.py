import re
import polars as pl
from typing import Callable
from dataclasses import dataclass, field

from blendsql.common.logger import logger, Color


@dataclass(frozen=True)
class LazyTable:
    """A non-materialized reference to a table.
    Allows us to delay execution of things like materialized a CTE
    until we actually interact with that table.
    """

    collect_fn: Callable[..., pl.DataFrame | pl.LazyFrame] = field()
    has_blendsql_function: bool = field()
    tablename: str | None = field(default=None)

    def __str__(self):
        return self.tablename or "N.A."

    def collect(self):
        if self.tablename is not None:
            logger.debug(Color.update(f"Materializing CTE `{self.tablename}`..."))
        return self.collect_fn()


class LazyTables(dict):
    """Used for storing LazyTable objects.

    Examples:
        ```python
        lazy_tables = LazyTables()
        # Add a LazyTable object
        lazy_tables.add(LazyTable("my_table", lambda _: pd.DataFrame())
        # Retrieve using tablename and call `collect()`
        df = lazy_tables.pop("my_table").collect()
        ```
    """

    def add(self, lazy_table: LazyTable):
        self[lazy_table.tablename] = lazy_table


def single_quote_escape(s):
    if "'" not in s:
        return s
    return re.sub(r"(?<!')'(?!')", "''", s)


def double_quote_escape(s):
    if '"' not in s:
        return s
    return re.sub(r'(?<!")"(?!")', '""', s)


def escape(s):
    return single_quote_escape(double_quote_escape(s))


def format_tuple(value: tuple, wrap_in_parentheses: bool | None = True):
    formatted = ", ".join(f"'{single_quote_escape(v)}'" for v in value)
    if wrap_in_parentheses:
        formatted = "(" + formatted + ")"
    return formatted


def select_all_from_table_query(tablename: str) -> str:
    return f'SELECT * FROM "{double_quote_escape(tablename)}";'


def truncate_df_content(df: pl.DataFrame, truncation_limit: int) -> pl.DataFrame:
    # Truncate long strings
    return df.with_columns(
        [
            pl.when(pl.col(col).str.len_chars() > truncation_limit)
            .then(pl.col(col).str.slice(0, truncation_limit) + "...")
            .otherwise(pl.col(col))
            .alias(col)
            for col in df.select(pl.col(pl.Utf8)).columns
        ]
    )
