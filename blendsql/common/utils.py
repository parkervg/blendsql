from typing import Tuple
from tabulate import tabulate
from functools import partial

from blendsql.common.constants import HF_REPO_ID, ColumnRef

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


def get_tablename_colname(s: ColumnRef) -> Tuple[str, str]:
    """Takes as input a string in the format `{tablename}.{colname}`
    Returns individual parts, but raises error if `s` is in the wrong format.
    """
    out = s.split(".")
    if len(out) != 2:
        raise ValueError(
            f"Invalid format: {s}\n" + "Expected format `{tablename}.{columnname}`"
        )
    tablename, colname = out
    return (tablename.strip('"'), colname.strip('"'))


def get_temp_subquery_table(
    session_uuid: str, subquery_idx: int, tablename: str
) -> str:
    """Generates temporary tablename for a subquery"""
    return f"{session_uuid}_{tablename}_{subquery_idx}"


def get_temp_session_table(session_uuid: str, tablename: str) -> str:
    """Generates temporary tablename for a BlendSQL execution session"""
    return f"{session_uuid}_{tablename}"
