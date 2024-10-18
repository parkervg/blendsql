"""Configure global BlendSQL parameters."""
import os

ASYNC_LIMIT_KEY = "BLENDSQL_ASYNC_LIMIT"
DEFAULT_ASYNC_LIMIT = "10"


def set_async_limit(n: int):
    os.environ[ASYNC_LIMIT_KEY] = str(n)
