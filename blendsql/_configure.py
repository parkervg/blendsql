"""Configure global BlendSQL parameters."""
import os

ASYNC_LIMIT_KEY = "BLENDSQL_ASYNC_LIMIT"
DEFAULT_ASYNC_LIMIT = "10"

MAX_OPTIONS_IN_PROMPT_KEY = "MAX_OPTIONS_IN_PROMPT"
DEFAULT_MAX_OPTIONS_IN_PROMPT = 50


def set_async_limit(n: int):
    os.environ[ASYNC_LIMIT_KEY] = str(n)


def set_max_options_in_prompt(n: int):
    os.environ[MAX_OPTIONS_IN_PROMPT_KEY] = str(n)
