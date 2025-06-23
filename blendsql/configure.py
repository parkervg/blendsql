"""Configure global BlendSQL parameters."""
import os

GLOBAL_HISTORY = []
MAX_HISTORY_SIZE = 20

ASYNC_LIMIT_KEY = "BLENDSQL_ASYNC_LIMIT"
DEFAULT_ASYNC_LIMIT = "10"

MAX_OPTIONS_IN_PROMPT_KEY = "MAX_OPTIONS_IN_PROMPT"
DEFAULT_MAX_OPTIONS_IN_PROMPT = 50


def add_to_global_history(entry: str):
    if len(GLOBAL_HISTORY) >= MAX_HISTORY_SIZE:
        GLOBAL_HISTORY.pop(0)

    GLOBAL_HISTORY.append(entry)


def set_async_limit(n: int):
    os.environ[ASYNC_LIMIT_KEY] = str(n)


def set_max_options_in_prompt(n: int):
    os.environ[MAX_OPTIONS_IN_PROMPT_KEY] = str(n)
