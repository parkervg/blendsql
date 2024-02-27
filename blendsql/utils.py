from typing import Tuple, Collection
import re
import logging
from colorama import Fore
from tabulate import tabulate
from functools import partial


from .db.sqlite import SQLite
from ._constants import HF_REPO_ID

tabulate = partial(tabulate, headers="keys", showindex="never", tablefmt="orgtbl")


def fetch_from_hub(filename: str):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            f"You need huggingface_hub to run this!\n`pip install huggingface_hub`"
        ) from None
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename, repo_type="dataset")


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


def delete_session_tables(db: SQLite, cleanup_tables: Collection[str]):
    """Deletes the temporary tables made for the sake of a BlendSQL execution session.

    Args:
        db: Database connector
        session_uuid: Unique string we used to identify temporary tables make during a BlendSQL session
    """
    if len(cleanup_tables) > 0:
        logging.debug(
            Fore.LIGHTMAGENTA_EX + f"Deleting temporary tables..." + Fore.RESET
        )
        for tablename in cleanup_tables:
            logging.debug(Fore.MAGENTA + f"Deleting {tablename}..." + Fore.RESET)
            db.con.execute(f"DROP TABLE '{tablename}'")


def recover_blendsql(select_sql: str):
    """Given a SQL `SELECT` statement, recovers BlendSQL syntax from SQLGlot SQLiteDialect interpretation.
    TODO: this is hack to convert sqlglot SQLite to BlendSQL.
    Examples:
        >>> recover_blendsql("STRUCT(STRUCT(QA('can i get my car fixed here?', 'transactions::merchant')))")
        {{QA('can i get my car fixed here?', 'transactions::merchant')}}
    """
    recovered = re.sub(
        r"(STRUCT\( ?STRUCT\()(.*?)(\)\)([^\)]|$))(,)?", r" {{\2}}\4 ", select_sql
    )
    # TODO: below is a hack for MIN(), MAX() type wrappings around an ingredient
    return re.sub(re.escape("))}}"), ")}})", recovered)


def get_temp_subquery_table(
    session_uuid: str, subquery_idx: str, tablename: str
) -> str:
    """Generates temporary tablename for a subquery"""
    return f"{session_uuid}_{tablename}_{subquery_idx}"


def get_temp_session_table(session_uuid: str, tablename: str) -> str:
    """Generates temporary tablename for a BlendSQL execution session"""
    return f"{session_uuid}_{tablename}"
