from typing import Tuple
import re
from tabulate import tabulate
from functools import partial

from ._constants import HF_REPO_ID

tabulate = partial(
    tabulate, headers="keys", showindex="never", tablefmt="simple_outline"
)
newline_dedent = lambda x: "\n".join([m.lstrip() for m in x.split("\n")])


def fetch_from_hub(filename: str):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            f"You need huggingface_hub to run this!\n`pip install huggingface_hub`"
        ) from None
    return hf_hub_download(
        repo_id=HF_REPO_ID, filename=filename, repo_type="dataset", force_download=False
    )


def get_tablename_colname(s: str) -> Tuple[str, str]:
    """Takes as input a string in the format `{tablename}::{colname}`
    Returns individual parts, but raises error if `s` is in the wrong format.
    """
    out = s.split("::")
    if len(out) != 2:
        raise ValueError(
            f"Invalid format: {s}\n" + "Expected format `{tablename}::{columnname}`"
        )
    tablename, colname = out
    return (tablename.strip('"'), colname.strip('"'))


def sub_tablename(original_tablename: str, new_tablename: str, query: str) -> str:
    """Replaces old tablename with a new tablename reference, likely one from a `get_temp` function.

    Args:
        original_tablename: String of the tablename in the current query to replace
        new_tablename: String of the new tablename
        query: BlendSQL query to do replacement in

    Returns:
        updated_query: BlendSQL query with tablenames subbed
    """
    return re.sub(
        # Only sub if surrounded by: whitespace, comma, or parentheses
        # Or, prefaced by period (e.g. 'p.Current_Value')
        r"(?<=( |,|\()|\.)\"?{}\"?(?=( |,|\)|;|\.|$))".format(original_tablename),
        new_tablename,
        query,
        flags=re.IGNORECASE,
    )


def recover_blendsql(select_sql: str):
    """Given a SQL `SELECT` statement, recovers BlendSQL syntax from SQLGlot SQLiteDialect interpretation.
    TODO: this is hack to convert sqlglot SQLite to BlendSQL.
    Examples:
        >>> recover_blendsql("STRUCT(STRUCT(QA('can i get my car fixed here?', 'transactions::merchant')))")
        {{QA('can i get my car fixed here?', 'transactions::merchant')}}
    """
    recovered = re.sub(
        r"(STRUCT\( ?STRUCT\()(.*?)(\){3})(,)?", r" {{\2)}}\4 ", select_sql
    )
    return recovered


def get_temp_subquery_table(
    session_uuid: str, subquery_idx: int, tablename: str
) -> str:
    """Generates temporary tablename for a subquery"""
    return f"{session_uuid}_{tablename}_{subquery_idx}"


def get_temp_session_table(session_uuid: str, tablename: str) -> str:
    """Generates temporary tablename for a BlendSQL execution session"""
    return f"{session_uuid}_{tablename}"
