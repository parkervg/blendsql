import re
import pandas as pd
from typing import Callable
from attr import attrs, attrib


@attrs(frozen=True)
class LazyTable:
    """A non-materialized reference to a table.
    Allows us to delay execution of things like materialized a CTE
    until we actually interact with that table.
    """

    tablename: str = attrib()
    collect: Callable[..., pd.DataFrame] = attrib()

    def __str__(self):
        return self.tablename


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
    return re.sub(r"(?<=[^'])'(?=[^'])", "''", s)


def double_quote_escape(s):
    return re.sub(r'(?<=[^"])"(?=[^"])', '""', s)


def escape(s):
    return single_quote_escape(double_quote_escape(s))


def select_all_from_table_query(tablename: str) -> str:
    return f'SELECT * FROM "{double_quote_escape(tablename)}";'


def truncate_df_content(df: pd.DataFrame, truncation_limit: int) -> pd.DataFrame:
    # Truncate long strings
    return df.map(
        lambda x: (
            f"{str(x)[:truncation_limit]}..."
            if isinstance(x, str) and len(str(x)) > truncation_limit
            else x
        )
    )
