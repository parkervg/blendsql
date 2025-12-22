import logging
import argparse
import litellm
import polars as pl
import pandas as pd
from dataclasses import asdict
from io import StringIO

from blendsql.common.logger import logger

from src.config import DUCKDB_DB_PATH, OUTPUT_DIR, MODEL_CONFIGS
from src.database_utils import create_duckdb_database
from src.eval_scripts import (
    run_blendsql_eval,
    run_thalamusdb_eval,
    run_flock_eval,
    run_lotus_eval,
)
from src.create_ground_truth import create_ground_truth
from src.evaluation.evaluate import MovieEvaluator

# Configure litellm
litellm.drop_params = True  # to disable `reasoning_effort` error in thalamusdb

logger.setLevel(logging.DEBUG)


def parse_args():
    """
    Parse command line arguments for selecting which evaluation systems to run.

    Returns:
        dict: Dictionary with system names as keys and boolean values
    """
    parser = argparse.ArgumentParser(
        description="Run movie database evaluation systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -bsf          # Run blendsql, flock
  python main.py -btl          # Run blendsql, thalamusdb, lotus
  python main.py -all          # Run all systems
  python main.py               # Run systems specified in config.py

System codes:
  b = blendsql
  f = flock
  t = thalamusdb
  l = lotus
        """,
    )

    parser.add_argument(
        "systems",
        nargs="?",
        type=str,
        default=None,
        help='System codes to run (e.g., "-bsf" or "-all"). If not provided, uses config.py settings.',
    )

    parser.add_argument(
        "lm_size", nargs="?", type=str, default=None, help="Gemma3 model size to run"
    )

    args = parser.parse_args()

    # Map of single-character codes to system names
    # we are lucky everyone has unique first letters
    system_map = {
        "b": "blendsql",
        "f": "flock",
        "t": "thalamusdb",
        "l": "lotus",
        "p": "palimpzest",
    }

    # Handle -all flag
    systems_str = args.systems.lstrip("-")
    if systems_str == "all":
        return {system: True for system in system_map.values()}

    # Parse individual system codes
    selected_systems = {system: False for system in system_map.values()}

    for char in systems_str:
        char_lower = char.lower()
        if char_lower in system_map:
            selected_systems[system_map[char_lower]] = True
        else:
            valid_codes = ", ".join(system_map.keys())
            parser.error(
                f"Invalid system code '{char}'. Valid codes are: {valid_codes}, or 'all'"
            )

    # Check if at least one system is selected
    if not any(selected_systems.values()):
        parser.error(
            "No valid systems selected. Please specify at least one system code."
        )

    return selected_systems, MODEL_CONFIGS[args.lm_size]


def main():
    """
    Main execution function.

    Run via `python main.py bt 4b` (where `bt` refers to the systems to evaluate and `12b` is model size)
    """
    evals_to_run, model_config = parse_args()

    # Display which systems will run
    enabled_systems = [name for name, enabled in evals_to_run.items() if enabled]
    print(f"Running evaluations for: {', '.join(enabled_systems)}\n")

    if not OUTPUT_DIR.is_dir():
        OUTPUT_DIR.mkdir(parents=True)

    if not DUCKDB_DB_PATH.is_file():
        create_duckdb_database()

    # Create ground truth
    evaluator = MovieEvaluator()
    ground_truth_results_df = create_ground_truth()
    ground_truth_results_df.to_csv(OUTPUT_DIR / "ground_truth_results.csv", index=False)

    all_results = []
    # Run enabled evaluations
    if evals_to_run.get("flock"):
        create_duckdb_database()
        flock_results_df = run_flock_eval(model_config)
        print(pl.DataFrame(flock_results_df))
        print(f"flock took an average of {flock_results_df['latency'].mean()} seconds")
        flock_results_df.to_csv(OUTPUT_DIR / "flock_results.csv", index=False)
        all_results.append(flock_results_df)

    if evals_to_run.get("blendsql"):
        create_duckdb_database()
        blendsql_results_df = run_blendsql_eval(model_config)
        print(pl.DataFrame(blendsql_results_df))
        print(
            f"blendsql took an average of {blendsql_results_df['latency'].mean()} seconds"
        )
        blendsql_results_df.to_csv(OUTPUT_DIR / "blendsql_results.csv", index=False)
        all_results.append(blendsql_results_df)

    if evals_to_run.get("thalamusdb"):
        create_duckdb_database()
        thalamusdb_results_df = run_thalamusdb_eval(model_config)
        print(pl.DataFrame(thalamusdb_results_df))
        print(
            f"thalamusdb took an average of {thalamusdb_results_df['latency'].mean()} seconds"
        )
        thalamusdb_results_df.to_csv(OUTPUT_DIR / "thalamusdb_results.csv", index=False)
        all_results.append(thalamusdb_results_df)

    if evals_to_run.get("lotus"):
        create_duckdb_database()
        lotus_results_df = run_lotus_eval(model_config)
        print(pl.DataFrame(lotus_results_df))
        print(f"lotus took an average of {lotus_results_df['latency'].mean()} seconds")
        lotus_results_df.to_csv(OUTPUT_DIR / "lotus_results.csv", index=False)
        all_results.append(lotus_results_df)

    all_results_df = pd.concat(all_results)
    all_results_df["metrics"] = None
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
            mask = (all_results_df["query_name"] == query_name) & (
                all_results_df["system_name"] == system_name
            )
            all_results_df.loc[mask, "metrics"] = [asdict(res)]

            print(system_name)
            print(asdict(res))
            print("\n\n")
        all_results_df.to_csv(OUTPUT_DIR / "all_results.csv")


if __name__ == "__main__":
    main()
