import logging
import argparse
import litellm
import polars as pl
import pandas as pd
from dataclasses import asdict
from io import StringIO
import time
import queue
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

from multiprocessing import Process, Queue
import signal
from contextlib import contextmanager

from blendsql.common.logger import logger

from src.config import DUCKDB_DB_PATH, MODEL_CONFIGS, ModelConfig
from src.database_utils import create_duckdb_database
from src.eval_scripts import (
    run_blendsql_eval,
    run_thalamusdb_eval,
    run_flock_eval,
    run_lotus_eval,
    run_palimpzest_eval,
)
from src.create_ground_truth import create_ground_truth
from src.evaluation.evaluate import MovieEvaluator

# Configure litellm
litellm.drop_params = True  # to disable `reasoning_effort` error in thalamusdb

logger.setLevel(logging.DEBUG)

# Timeout for each evaluation (in seconds) - adjust as needed
EVAL_TIMEOUT = 3600 * 3

# Number of runs per system
N_RUNS = 3  # Change this to your desired number of runs


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds):
    """Context manager for timing out operations (Unix only)"""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Only works on Unix systems
    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # On Windows, just yield without timeout
        yield


def parse_args() -> tuple[list[str], ModelConfig, int, int]:
    """
    Parse command line arguments for selecting which evaluation systems to run.

    Returns:
        tuple: (selected_systems, model_config, timeout, n_runs)
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

    parser.add_argument(
        "--timeout",
        type=int,
        default=EVAL_TIMEOUT,
        help=f"Timeout for each evaluation in seconds (default: {EVAL_TIMEOUT})",
    )

    parser.add_argument(
        "--n-runs",
        type=int,
        default=N_RUNS,
        help=f"Number of runs per system (default: {N_RUNS})",
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
        return (
            {system: True for system in system_map.values()},
            MODEL_CONFIGS[args.lm_size],
            args.timeout,
            args.n_runs,
        )

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

    return selected_systems, MODEL_CONFIGS[args.lm_size], args.timeout, args.n_runs


def run_eval_in_process(eval_func, model_config, system_name, run_number, output_queue):
    """
    Wrapper function to run an evaluation in a separate process.

    Args:
        eval_func: The evaluation function to run
        model_config: Model configuration to pass to the eval function
        system_name: Name of the system being evaluated
        run_number: The run number (1-indexed)
        output_queue: Queue to put results into
    """
    try:
        # Recreate database for this process
        create_duckdb_database()

        # Run the evaluation
        print(f"[{system_name}] Run {run_number}: Starting evaluation...")
        results_df = eval_func(model_config)

        # Add system_name and run columns
        results_df["system_name"] = system_name
        results_df["run"] = run_number

        print(f"[{system_name}] Run {run_number}: Evaluation complete!")

        # Convert to dict for serialization through queue
        results_dict = results_df.to_dict(orient="records")

        avg_latency = results_df["latency"].mean()

        # Put results in queue
        output_queue.put(
            {
                "system_name": system_name,
                "run": run_number,
                "results": results_dict,
                "success": True,
                "error": None,
                "avg_latency": avg_latency,
            }
        )

        import time

        time.sleep(0.3)  # Give queue time to flush

        print(
            f"[{system_name}] Run {run_number}: Average latency: {avg_latency:.2f} seconds"
        )

    except Exception as e:
        print(f"[{system_name}] Run {run_number} ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        output_queue.put(
            {
                "system_name": system_name,
                "run": run_number,
                "results": None,
                "success": False,
                "error": str(e),
                "avg_latency": None,
            }
        )


def main():
    """
    Main execution function.

    Run via `python main.py bt 4b` (where `bt` refers to the systems to evaluate and `12b` is model size)
    """
    from src.config import BASE_OUTPUT_DIR

    evals_to_run, model_config, eval_timeout, n_runs = parse_args()

    OUTPUT_DIR = BASE_OUTPUT_DIR / model_config.model_name_or_path.replace("/", "_")
    # Display which systems will run
    enabled_systems = [name for name, enabled in evals_to_run.items() if enabled]
    print(f"Running evaluations for: {', '.join(enabled_systems)}")
    print(f"Using model: {model_config.model_name_or_path}")
    print(f"Number of runs per system: {n_runs}")
    print(f"Timeout per evaluation: {eval_timeout} seconds\n")

    if not OUTPUT_DIR.is_dir():
        OUTPUT_DIR.mkdir(parents=True)

    if not DUCKDB_DB_PATH.is_file():
        create_duckdb_database()

    # Create ground truth
    print("Creating ground truth...")
    evaluator = MovieEvaluator()
    ground_truth_results_df = create_ground_truth()
    ground_truth_results_df.to_csv(OUTPUT_DIR / "ground_truth_results.csv", index=False)
    print("Ground truth created\n")

    # Map of system names to their evaluation functions
    eval_functions = {
        "flock": run_flock_eval,
        "blendsql": run_blendsql_eval,
        "thalamusdb": run_thalamusdb_eval,
        "lotus": run_lotus_eval,
        "palimpzest": run_palimpzest_eval,
    }

    all_results = []
    failed_runs = []  # Track failed runs with (system_name, run_number)

    # Run each enabled evaluation multiple times in separate processes
    for system_name, should_run in evals_to_run.items():
        if not should_run or system_name not in eval_functions:
            continue

        system_results = []  # Store results for this system across all runs

        for run_number in range(1, n_runs + 1):
            print(f"\n{'=' * 60}")
            print(f"Running {system_name} evaluation - Run {run_number}/{n_runs}")
            print(f"{'=' * 60}\n")

            # Create a queue to get results back from the process
            result_queue = Queue()

            # Create and start the process
            eval_func = eval_functions[system_name]
            process = Process(
                target=run_eval_in_process,
                args=(eval_func, model_config, system_name, run_number, result_queue),
                name=f"{system_name}_eval_run_{run_number}",
            )
            process.start()

            # Read from queue while waiting for process (with timeout)
            result = None
            start_time = time.time()
            while time.time() - start_time < eval_timeout:
                try:
                    result = result_queue.get(timeout=1)  # Check every second
                    break
                except queue.Empty:
                    if not process.is_alive():
                        # Process died without putting result
                        break

            # Now join should be instant (or process timed out)
            process.join(timeout=5)

            # Check if process is still alive (timed out)
            if process.is_alive():
                print(
                    f"\n[{system_name}] Run {run_number} WARNING: Evaluation timed out after {eval_timeout} seconds"
                )
                process.terminate()
                process.join(timeout=10)

                if process.is_alive():
                    process.kill()
                    process.join()

                failed_runs.append((system_name, run_number))
                continue

            # Process result
            if result is not None and result["success"]:
                results_df = pd.DataFrame(result["results"])
                print(f"\n[{system_name}] Run {run_number} Results:")
                print(
                    pl.DataFrame(
                        results_df[["query_name", "latency", "system_name", "run"]]
                    )
                )
                print(
                    f"\n[{system_name}] Run {run_number}: Average latency: {result['avg_latency']:.2f} seconds"
                )
                system_results.append(results_df)
                all_results.append(results_df)
            elif result is not None:
                print(f"[{system_name}] Run {run_number} ERROR: {result['error']}")
                failed_runs.append((system_name, run_number))
            else:
                print(
                    f"[{system_name}] Run {run_number} WARNING: No results received from process"
                )
                failed_runs.append((system_name, run_number))

        # Save all runs for this system to a single CSV
        if system_results:
            system_all_runs_df = pd.concat(system_results, ignore_index=True)
            output_file = OUTPUT_DIR / f"{system_name}_all_runs_results.csv"
            system_all_runs_df.to_csv(output_file, index=False)
            print(
                f"[{system_name}] Saved all {len(system_results)} runs to {output_file}"
            )

    # Report on failed runs
    if failed_runs:
        print("\n" + "!" * 60)
        print(f"WARNING: {len(failed_runs)} run(s) failed:")
        for system, run in failed_runs:
            print(f"  - {system} (run {run})")
        print("!" * 60 + "\n")

    # Combine and evaluate all results
    def extract_quality_metric(query_data: dict) -> float:
        """Extract and normalize quality metric from query data."""
        if "f1_score" in query_data:
            return min(1.0, max(0.0, query_data["f1_score"]))
        elif "spearman_correlation" in query_data:
            # Convert from [-1, 1] to [0, 1] range
            corr = query_data["spearman_correlation"]
            return (corr + 1) / 2
        elif "relative_error" in query_data:
            # Transform relative error to quality score (lower error = higher quality)
            error = query_data["relative_error"]
            # Cap error at 1.0 to prevent negative scores
            return max(0.0, 1.0 - min(1.0, error))
        else:
            print(f"Warning: No quality metric found in query data, using 0.0")
            return 0.0

    if all_results:
        print("\n" + "=" * 60)
        print("Combining and evaluating all results...")
        print("=" * 60 + "\n")

        all_results_df = pd.concat(all_results, ignore_index=True)
        all_results_df["metrics"] = None

        # Evaluate each query for each system and run
        for query_name in all_results_df["query_name"].unique():
            print(f"Evaluating {query_name}...")
            reference = pd.read_json(
                StringIO(
                    ground_truth_results_df[
                        ground_truth_results_df["query_name"] == query_name
                    ]["prediction"].item()
                ),
                orient="split",
            )

            for system_name in all_results_df["system_name"].unique():
                for run_number in all_results_df[
                    all_results_df["system_name"] == system_name
                ]["run"].unique():
                    _prediction = all_results_df[
                        (all_results_df["query_name"] == query_name)
                        & (all_results_df["system_name"] == system_name)
                        & (all_results_df["run"] == run_number)
                    ]
                    if _prediction.empty:
                        continue

                    prediction = pd.read_json(
                        StringIO(_prediction["prediction"].item()), orient="split"
                    )

                    metric_dict: dict = asdict(
                        evaluator.evaluate_single_query(
                            int(query_name.replace("Q", "")),
                            system_results=prediction,
                            ground_truth=reference,
                        )
                    )
                    mask = (
                        (all_results_df["query_name"] == query_name)
                        & (all_results_df["system_name"] == system_name)
                        & (all_results_df["run"] == run_number)
                    )
                    all_results_df.loc[mask, "raw_metrics"] = [metric_dict]
                    all_results_df.loc[mask, "quality"] = [
                        extract_quality_metric(metric_dict)
                    ]
                    print(f"  {system_name} (run {run_number}): {metric_dict}")

        # Save combined results
        all_results_df.to_csv(OUTPUT_DIR / "all_results_with_runs.csv", index=False)
        print(f"\nSaved combined results to {OUTPUT_DIR / 'all_results_with_runs.csv'}")
    else:
        print("\nNo results to combine (all evaluations failed)")


if __name__ == "__main__":
    main()
