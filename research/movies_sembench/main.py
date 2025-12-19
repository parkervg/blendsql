import logging

import litellm

from blendsql.common.logger import logger
import polars as pl

from src.config import EVALS_TO_RUN, DUCKDB_DB_PATH, OUTPUT_DIR
from src.model_utils import download_model_if_needed
from src.database_utils import create_duckdb_database
from src.eval_scripts import run_blendsql_eval, run_thalamusdb_eval, run_flock_eval

# Configure litellm
litellm.drop_params = True  # to disable `reasoning_effort` error in thalamusdb

logger.setLevel(logging.DEBUG)


def main():
    """
    Main execution function.

    1. Downloads model if needed
    2. Loads gold standard results
    3. Runs enabled evaluations
    """
    if not OUTPUT_DIR.is_dir():
        OUTPUT_DIR.mkdir(parents=True)

    # Download GGUF model if not exists
    if not download_model_if_needed():
        print("Failed to download model. Exiting.")
        return

    if not DUCKDB_DB_PATH.is_file():
        create_duckdb_database()

    # Run enabled evaluations
    if EVALS_TO_RUN.get("flock"):
        flock_results_df = run_flock_eval()
        print(pl.DataFrame(flock_results_df))
        system_output_dir = OUTPUT_DIR / "flock"
        if not system_output_dir.is_dir():
            system_output_dir.mkdir(parents=True)
        flock_results_df.to_csv(system_output_dir / "results.csv", index=False)

    if EVALS_TO_RUN.get("blendsql"):
        blendsql_results_df = run_blendsql_eval()
        print(pl.DataFrame(blendsql_results_df))
        OUTPUT_DIR / "blendsql/results.csv"
        system_output_dir = OUTPUT_DIR / "blendsql"
        if not system_output_dir.is_dir():
            system_output_dir.mkdir(parents=True)
        blendsql_results_df.to_csv(system_output_dir / "results.csv", index=False)

    if EVALS_TO_RUN.get("thalamusdb"):
        thalamusdb_results_df = run_thalamusdb_eval()
        print(pl.DataFrame(thalamusdb_results_df))
        system_output_dir = OUTPUT_DIR / "thalamusdb"
        if not system_output_dir.is_dir():
            system_output_dir.mkdir(parents=True)
        thalamusdb_results_df.to_csv(system_output_dir / "results.csv", index=False)


if __name__ == "__main__":
    main()
