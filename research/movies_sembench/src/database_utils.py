import duckdb
import pandas as pd
from typing import Generator
import logging

from blendsql.common.logger import Color, logger
from blendsql.common.utils import fetch_from_hub

from src.config import (
    DUCKDB_DB_PATH,
    SKIP_QUERIES,
    ONLY_USE,
    QUERIES_DIR,
    USE_DATA_SIZE,
)

logger.setLevel(logging.DEBUG)


def create_duckdb_database():
    """
    Create or recreate the DuckDB database with movie data.

    Not sure if other methods modify database state - so we want to reset before each eval.
    """
    logger.debug(Color.update("Creating database..."))

    if DUCKDB_DB_PATH.is_file():
        DUCKDB_DB_PATH.unlink()

    conn = duckdb.connect(DUCKDB_DB_PATH)

    # Load data from hub
    movies_df = pd.read_csv(fetch_from_hub(f"movie/sf_{USE_DATA_SIZE}/Movies.csv"))
    reviews_df = pd.read_csv(fetch_from_hub(f"movie/sf_{USE_DATA_SIZE}/Reviews.csv"))

    # Register DataFrames
    conn.register("movies_df", movies_df)
    conn.register("reviews_df", reviews_df)

    # Create persistent tables
    conn.execute("CREATE OR REPLACE TABLE Movies AS SELECT * FROM movies_df")
    conn.execute("CREATE OR REPLACE TABLE Reviews AS SELECT * FROM reviews_df")

    logger.debug(Color.success("Database tables created successfully"))
    conn.close()


def load_gold_standard_results() -> dict[str, pd.DataFrame]:
    """
    Load gold standard SQL query results.

    Returns:
        Dictionary mapping query names to their result DataFrames
    """
    query_name_to_gold: dict[str, pd.DataFrame] = {}

    conn = duckdb.connect(DUCKDB_DB_PATH, read_only=True)

    for query_file, query_name in iter_queries("gold_sql"):
        res = conn.execute(open(query_file).read()).df()
        print(res)
        query_name_to_gold[query_name] = res

    conn.close()
    return query_name_to_gold


def iter_queries(system_name: str) -> Generator:
    """
    Iterate through query files for a given system.

    Args:
        system_name: Name of the system directory containing queries

    Yields:
        Tuples of (query_file_path, query_name)
    """
    queries_path = QUERIES_DIR / system_name
    sorted_query_files = sorted(queries_path.iterdir(), key=lambda x: x.stem)

    for query_file in sorted_query_files:
        query_name = query_file.stem
        if query_name in SKIP_QUERIES:
            continue
        if ONLY_USE and query_name not in ONLY_USE:
            continue
        print(f"Running {system_name} {query_name}...")
        yield (query_file, query_name)
