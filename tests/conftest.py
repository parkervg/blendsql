from blendsql.db import SQLite, DuckDB


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, SQLite):
        return f"SQLite"
    if isinstance(val, DuckDB):
        return f"DuckDB"
    # return None to let pytest handle the formatting
    return None
