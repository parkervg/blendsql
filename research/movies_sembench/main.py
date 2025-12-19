import logging

import litellm

from blendsql.common.logger import logger

from src.config import EVALS_TO_RUN, DUCKDB_DB_PATH
from src.model_utils import download_model_if_needed
from src.database_utils import load_gold_standard_results, create_duckdb_database
from src.eval_scripts import run_blendsql_eval, run_thalamusdb_eval, run_flock_eval

# Configure litellm
litellm.drop_params = True  # to disable `reasoning_effort` error in thalamusdb

# Set logging level
logger.setLevel(logging.DEBUG)


class DummyConsole:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


import rich.console

rich.console.Console = DummyConsole


def main():
    """
    Main execution function.

    1. Downloads model if needed
    2. Loads gold standard results
    3. Runs enabled evaluations
    """
    # Download GGUF model if not exists
    if not download_model_if_needed():
        print("Failed to download model. Exiting.")
        return

    if not DUCKDB_DB_PATH.is_file():
        create_duckdb_database()

    # Load gold standard results for comparison
    load_gold_standard_results()

    # Run enabled evaluations
    if EVALS_TO_RUN.get("flock"):
        run_flock_eval()

    if EVALS_TO_RUN.get("blendsql"):
        run_blendsql_eval()

    if EVALS_TO_RUN.get("thalamusdb"):
        run_thalamusdb_eval()


if __name__ == "__main__":
    main()
