from typing import Tuple, Iterable
import re
from pathlib import Path
import json
import os
import logging
from colorama import Fore
import time
import threading

from .db.sqlite_db_connector import SQLiteDBConnector


def init_secrets(filepath: str) -> None:
    """Initializes environment variables from secrets.json

    Args:
        path: Filepath to JSON file with keys/values of secrets
    """
    filepath = Path(filepath)
    if filepath.suffix != ".json":
        raise ValueError(
            f"init_secrets argument should be a JSON filepath\nGot{filepath}"
        )
    # Initialize env variables from secrets.json
    if (filepath).is_file():
        with open(filepath, "r") as f:
            secrets = json.load(f)
        for k, v in secrets.items():
            os.environ[k] = v
        if secrets.get("OPENAI_API_KEY", None) is None:
            try:
                from azure.identity import ClientSecretCredential

                credential = ClientSecretCredential(
                    tenant_id=secrets["TENANT_ID"],
                    client_id=secrets["CLIENT_ID"],
                    client_secret=secrets["CLIENT_SECRET"],
                    disable_instance_discovery=True,
                )
                access_token = credential.get_token(
                    secrets["TOKEN_SCOPE"],
                    tenant_id=secrets["TENANT_ID"],
                )
                os.environ["OPENAI_API_KEY"] = access_token.token
            except KeyError:
                raise ValueError(
                    f"Error authenticating with OpenAI\n Without explicit `OPENAI_API_KEY`, you need to provide ['TENANT_ID', 'CLIENT_ID', 'CLIENT_SECRET']"
                ) from None
    else:
        raise ValueError(f"Can't find secrets.json at {filepath}")


class TokenTimer(threading.Thread):
    """Class to handle refreshing OpenAI tokens."""

    def __init__(self, secrets_json_path: str, refresh_interval_min: int = 30):
        super().__init__()
        self.daemon = True
        self.secrets_json_path = secrets_json_path
        self.refresh_interval_min = refresh_interval_min

    def run(self):
        while True:
            print(Fore.YELLOW + f"Refreshing our OpenAI tokens..." + Fore.RESET)
            init_secrets(self.secrets_json_path)
            time.sleep(self.refresh_interval_min * self.refresh_interval_min)


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


def delete_session_tables(db: SQLiteDBConnector, cleanup_tables: Iterable[str]):
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
    Example:
        STRUCT(STRUCT(QA('can i get my car fixed here?', 'transactions::merchant'))) -> {{QA('can i get my car fixed here?', 'transactions::merchant')}}
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
