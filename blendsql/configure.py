"""Configure global BlendSQL parameters."""
import os

GLOBAL_HISTORY = []
MAX_HISTORY_SIZE = 20

_default_model = None

ASYNC_LIMIT_KEY = "BLENDSQL_ASYNC_LIMIT"
DEFAULT_ASYNC_LIMIT = 32

MAX_OPTIONS_IN_PROMPT_KEY = "MAX_OPTIONS_IN_PROMPT"
DEFAULT_MAX_OPTIONS_IN_PROMPT = 50

MAX_TOKENS_KEY = "BLENDSQL_MAX_TOKENS"
DEFAULT_MAX_TOKENS = 4096

DETERMINISTIC_KEY = "BLENDSQL_DETERMINISTIC"
DEFAULT_DETERMINISTIC = 0


def add_to_global_history(entry: str):
    if len(GLOBAL_HISTORY) >= MAX_HISTORY_SIZE:
        GLOBAL_HISTORY.pop(0)

    GLOBAL_HISTORY.append(entry)


def set_async_limit(n: int):
    os.environ[ASYNC_LIMIT_KEY] = str(n)


def set_max_options_in_prompt(n: int):
    os.environ[MAX_OPTIONS_IN_PROMPT_KEY] = str(n)


def set_default_max_tokens(n: int):
    os.environ[MAX_TOKENS_KEY] = str(n)


def set_deterministic(v: bool):
    os.environ[DETERMINISTIC_KEY] = str(int(v))


class _Config:
    def __call__(self, model=None):
        global _default_model
        if model is not None:
            _default_model = model

    def set_async_limit(self, n: int):
        set_async_limit(n)

    def set_max_options_in_prompt(self, n: int):
        set_max_options_in_prompt(n)

    def set_default_max_tokens(self, n: int):
        set_default_max_tokens(n)

    def set_deterministic(self, v: bool):
        set_deterministic(v)


config = _Config()
