import logging

import litellm

from blendsql.common.logger import logger
import polars as pl
import pandas as pd
from dataclasses import asdict
from io import StringIO

from src.config import EVALS_TO_RUN, DUCKDB_DB_PATH, OUTPUT_DIR
from src.model_utils import download_model_if_needed
from src.database_utils import create_duckdb_database
from src.eval_scripts import run_blendsql_eval, run_thalamusdb_eval, run_flock_eval
from src.create_ground_truth import create_ground_truth
from src.evaluation.evaluate import MovieEvaluator

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

    # Create ground truth
    evaluator = MovieEvaluator()
    ground_truth_results_df = create_ground_truth()
    ground_truth_results_df.to_csv(OUTPUT_DIR / "ground_truth_results.csv", index=False)

    all_results = []
    # Run enabled evaluations
    if EVALS_TO_RUN.get("flock"):
        flock_results_df = run_flock_eval()
        print(pl.DataFrame(flock_results_df))
        print(f"flock took an average of {flock_results_df['latency'].mean()} seconds")
        flock_results_df.to_csv(OUTPUT_DIR / "flock_results.csv", index=False)
        all_results.append(flock_results_df)

    if EVALS_TO_RUN.get("blendsql"):
        blendsql_results_df = run_blendsql_eval()
        print(pl.DataFrame(blendsql_results_df))
        print(
            f"blendsql took an average of {blendsql_results_df['latency'].mean()} seconds"
        )
        blendsql_results_df.to_csv(OUTPUT_DIR / "blendsql_results.csv", index=False)
        all_results.append(blendsql_results_df)

    if EVALS_TO_RUN.get("thalamusdb"):
        thalamusdb_results_df = run_thalamusdb_eval()
        print(pl.DataFrame(thalamusdb_results_df))
        print(
            f"thalamusdb took an average of {thalamusdb_results_df['latency'].mean()} seconds"
        )
        thalamusdb_results_df.to_csv(OUTPUT_DIR / "thalamusdb_results.csv", index=False)
        all_results.append(thalamusdb_results_df)

    all_results_df = pd.concat(all_results)
    for query_name in all_results_df["query_name"].unique():
        print(query_name)
        reference = pd.read_json(
            StringIO(
                ground_truth_results_df[
                    ground_truth_results_df["query_name"] == query_name
                ]["prediction"].item()
            ),
            orient="split",
        )
        for system_name in all_results_df["system_name"].unique():
            _prediction = all_results_df[
                (all_results_df["query_name"] == query_name)
                & (all_results_df["system_name"] == system_name)
            ]
            if _prediction.empty:
                continue
            prediction = pd.read_json(
                StringIO(_prediction["prediction"].item()), orient="split"
            )
            res = evaluator.evaluate_single_query(
                int(query_name.replace("Q", "")),
                system_results=prediction,
                ground_truth=reference,
            )
            print(system_name)
            print(asdict(res))
            print("\n\n")


if __name__ == "__main__":
    main()
