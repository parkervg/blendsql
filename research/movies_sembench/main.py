import logging
import argparse
import litellm
import polars as pl
import pandas as pd
from dataclasses import asdict
from io import StringIO
from multiprocessing import Process, Queue
import signal
from contextlib import contextmanager

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

# Timeout for each evaluation (in seconds) - adjust as needed
EVAL_TIMEOUT = 3600  # 1 hour


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

    parser.add_argument(
        "--timeout",
        type=int,
        default=EVAL_TIMEOUT,
        help=f"Timeout for each evaluation in seconds (default: {EVAL_TIMEOUT})",
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

    return selected_systems, MODEL_CONFIGS[args.lm_size], args.timeout


def run_eval_in_process(eval_func, model_config, system_name, output_queue):
    """
    Wrapper function to run an evaluation in a separate process.

    Args:
        eval_func: The evaluation function to run
        model_config: Model configuration to pass to the eval function
        system_name: Name of the system being evaluated
        output_queue: Queue to put results into
    """
    try:
        # Recreate database for this process
        create_duckdb_database()

        # Run the evaluation
        print(f"[{system_name}] Starting evaluation...")
        results_df = eval_func(model_config)
        print(f"[{system_name}] Evaluation complete!")

        # Convert to dict for serialization through queue
        results_dict = results_df.to_dict(orient="records")

        avg_latency = results_df["latency"].mean()

        # Put results in queue
        output_queue.put(
            {
                "system_name": system_name,
                "results": results_dict,
                "success": True,
                "error": None,
                "avg_latency": avg_latency,
            }
        )

        print(f"[{system_name}] Average latency: {avg_latency:.2f} seconds")

    except Exception as e:
        print(f"[{system_name}] ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        output_queue.put(
            {
                "system_name": system_name,
                "results": None,
                "success": False,
                "error": str(e),
                "avg_latency": None,
            }
        )
    finally:
        # Aggressive cleanup
        import gc

        gc.collect()

        # Try to clean up any CUDA memory
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass


def main():
    """
    Main execution function.

    Run via `python main.py bt 4b` (where `bt` refers to the systems to evaluate and `12b` is model size)
    """
    evals_to_run, model_config, eval_timeout = parse_args()

    # Display which systems will run
    enabled_systems = [name for name, enabled in evals_to_run.items() if enabled]
    print(f"Running evaluations for: {', '.join(enabled_systems)}")
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
    }

    all_results = []
    failed_systems = []

    # Run each enabled evaluation in a separate process
    for system_name, should_run in evals_to_run.items():
        if not should_run or system_name not in eval_functions:
            continue

        print(f"\n{'=' * 60}")
        print(f"Running {system_name} evaluation in separate process...")
        print(f"{'=' * 60}\n")

        # Create a queue to get results back from the process
        result_queue = Queue()

        # Create and start the process
        eval_func = eval_functions[system_name]
        process = Process(
            target=run_eval_in_process,
            args=(eval_func, model_config, system_name, result_queue),
            name=f"{system_name}_eval",
        )
        process.start()

        # Wait for the process to complete with timeout
        process.join(timeout=eval_timeout)

        # Check if process is still alive (timed out)
        if process.is_alive():
            print(
                f"\n[{system_name}] WARNING: Evaluation timed out after {eval_timeout} seconds"
            )
            print(f"[{system_name}] Terminating process...")
            process.terminate()
            process.join(timeout=10)  # Give it 10 seconds to terminate gracefully

            if process.is_alive():
                print(f"[{system_name}] Force killing process...")
                process.kill()
                process.join()

            failed_systems.append(system_name)
            print(f"[{system_name}] Process terminated due to timeout\n")
            continue

        # Get results from queue
        if not result_queue.empty():
            result = result_queue.get()

            if result["success"]:
                # Convert back to DataFrame
                results_df = pd.DataFrame(result["results"])

                # Print results
                print(f"\n[{system_name}] Results:")
                print(pl.DataFrame(results_df))
                print(
                    f"\n[{system_name}] Average latency: {result['avg_latency']:.2f} seconds"
                )

                # Save to CSV
                output_file = OUTPUT_DIR / f"{system_name}_results.csv"
                results_df.to_csv(output_file, index=False)
                print(f"[{system_name}] Saved results to {output_file}")

                all_results.append(results_df)
            else:
                print(f"[{system_name}] ERROR: {result['error']}")
                failed_systems.append(system_name)
        else:
            print(f"[{system_name}] WARNING: No results received from process")
            failed_systems.append(system_name)

        # Check if process exited cleanly
        if process.exitcode != 0:
            print(
                f"[{system_name}] WARNING: Process exited with code {process.exitcode}"
            )
            if system_name not in failed_systems:
                failed_systems.append(system_name)

        print(f"\n[{system_name}] Process completed and memory cleaned up\n")

    # Report on failed systems
    if failed_systems:
        print("\n" + "!" * 60)
        print(f"WARNING: {len(failed_systems)} system(s) failed:")
        for system in failed_systems:
            print(f"  - {system}")
        print("!" * 60 + "\n")

    # Combine and evaluate all results
    if all_results:
        print("\n" + "=" * 60)
        print("Combining and evaluating all results...")
        print("=" * 60 + "\n")

        all_results_df = pd.concat(all_results, ignore_index=True)
        all_results_df["metrics"] = None

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

                print(f"  {system_name}: {asdict(res)}\n")

        # Save combined results
        all_results_df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
        print(f"\nSaved combined results to {OUTPUT_DIR / 'all_results.csv'}")
        print("\nFinal Results:")
        print(pl.DataFrame(all_results_df))

        # Print summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        summary = (
            all_results_df.groupby("system_name")
            .agg({"latency": ["mean", "std", "min", "max"]})
            .round(2)
        )
        print(summary)
    else:
        print("\nNo results to combine (all evaluations failed)")


if __name__ == "__main__":
    main()
