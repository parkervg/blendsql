import re
import pandas as pd


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
