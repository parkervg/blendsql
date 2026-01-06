import sys
import gc
import subprocess

from src.config import (
    MODEL_CONFIGS,
    MODEL_PARAMS,
)
from src.server_utils import (
    start_llama_cpp_server,
    stop_llama_cpp_server,
    LLAMA_SERVER_BASE_URL,
    LLAMA_SERVER_COMPLETION_ENDPOINT,
)

import time
import statistics
from typing import List, Dict, Tuple, Optional

from llama_cpp import Llama
from transformers import AutoTokenizer
import requests

BENCHMARK_PROMPT = """You are a helpful assistant.
Explain the theory of relativity in simple terms, with LOTS of examples.
"""
BENCHMARK_RUNS = 100
BENCHMARK_TEMPERATURE = 0.7
MAX_TOKENS = 2

MODEL_CONFIG = MODEL_CONFIGS["12b"]

TOKENIZER_PATH = "google/gemma-3-4b-it"

BENCHMARK_LLAMA_CPP_CONFIG = {
    "n_gpu_layers": -1,
    "n_ctx": MODEL_PARAMS["num_ctx"],
    "seed": MODEL_PARAMS["seed"],
    "n_threads": MODEL_PARAMS["num_threads"],
    "chat_format": MODEL_CONFIG.chat_format,
    "flash_attn": MODEL_PARAMS["flash_attn"],
}

# Global variable to track server process
_llama_server_process: Optional[subprocess.Popen] = None


def tokens_per_second(token_count: int, elapsed: float) -> float:
    """
    Calculate tokens per second.

    Args:
        token_count: Number of tokens generated
        elapsed: Time elapsed in seconds

    Returns:
        Tokens per second
    """
    if elapsed == 0:
        return 0.0
    return token_count / elapsed


def benchmark_local_llama_cpp(
    prompt: str = BENCHMARK_PROMPT,
    max_tokens: int = MAX_TOKENS,
    runs: int = BENCHMARK_RUNS,
    temperature: float = BENCHMARK_TEMPERATURE,
    verbose: bool = True,
) -> Tuple[List[float], Dict[str, float]]:
    """
    Benchmark local llama-cpp-python inference.

    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        runs: Number of benchmark runs
        temperature: Sampling temperature
        verbose: Print progress

    Returns:
        Tuple of (speeds list, stats dict)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("=== Benchmarking Local llama-cpp-python ===")
        print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    llm = Llama.from_pretrained(
        repo_id=str(MODEL_CONFIG.model_name_or_path),
        filename=str(MODEL_CONFIG.filename),
        **BENCHMARK_LLAMA_CPP_CONFIG,
    )

    if verbose:
        print(f"\nRunning {runs} iterations...")
        print(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")
        print(f"Max tokens: {max_tokens}")
        print(f"Temperature: {temperature}\n")

    speeds = []

    for i in range(runs):
        start = time.perf_counter()

        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        elapsed = time.perf_counter() - start

        # Get generated text and count tokens
        text = output["choices"][0]["text"]
        token_count = len(tokenizer.encode(text))

        # Calculate speed
        speed = tokens_per_second(token_count, elapsed)
        speeds.append(speed)

        if verbose:
            print(
                f"Run {i + 1:2d}/{runs}: {token_count:3d} tokens in {elapsed:5.2f}s → {speed:6.2f} tok/s"
            )

    # Calculate statistics
    stats = {
        "mean": statistics.mean(speeds),
        "median": statistics.median(speeds),
        "stdev": statistics.stdev(speeds) if len(speeds) > 1 else 0.0,
        "min": min(speeds),
        "max": max(speeds),
    }

    if verbose:
        print("\n" + "-" * 60)
        print(f"Average:  {stats['mean']:.2f} tok/s")
        print(f"Median:   {stats['median']:.2f} tok/s")
        print(f"Std Dev:  {stats['stdev']:.2f} tok/s")
        print(f"Min:      {stats['min']:.2f} tok/s")
        print(f"Max:      {stats['max']:.2f} tok/s")
        print("=" * 60)
    if verbose:
        print("\nCleaning up llama-cpp model from memory...")

    # Delete the model object
    del llm

    # Delete tokenizer to free memory
    del tokenizer

    # Force garbage collection
    gc.collect()

    # Try to clear CUDA cache if available
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if verbose:
                print("✓ GPU cache cleared")
    except ImportError:
        pass

    if verbose:
        print("✓ Memory cleanup complete")

    return speeds, stats


def benchmark_llama_cpp_server(
    prompt: str = BENCHMARK_PROMPT,
    max_tokens: int = MAX_TOKENS,
    runs: int = BENCHMARK_RUNS,
    temperature: float = BENCHMARK_TEMPERATURE,
    verbose: bool = True,
) -> Tuple[List[float], Dict[str, float]]:
    """
    Benchmark llama-cpp-server inference.

    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        runs: Number of benchmark runs
        temperature: Sampling temperature
        verbose: Print progress

    Returns:
        Tuple of (speeds list, stats dict)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("=== Benchmarking llama-cpp-server ===")
        print("=" * 60)

    # Load tokenizer for counting
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    if verbose:
        print(f"Server URL: {LLAMA_SERVER_BASE_URL}")
        print(f"\nRunning {runs} iterations...")
        print(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")
        print(f"Max tokens: {max_tokens}")
        print(f"Temperature: {temperature}\n")

    speeds = []

    for i in range(runs):
        start = time.perf_counter()

        response = requests.post(
            LLAMA_SERVER_COMPLETION_ENDPOINT,
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
        )

        elapsed = time.perf_counter() - start

        # Get generated text and count tokens
        result = response.json()
        text = result["choices"][0]["text"]
        token_count = len(tokenizer.encode(text))

        # Calculate speed
        speed = tokens_per_second(token_count, elapsed)
        speeds.append(speed)

        if verbose:
            print(
                f"Run {i + 1:2d}/{runs}: {token_count:3d} tokens in {elapsed:5.2f}s → {speed:6.2f} tok/s"
            )

    # Calculate statistics
    stats = {
        "mean": statistics.mean(speeds),
        "median": statistics.median(speeds),
        "stdev": statistics.stdev(speeds) if len(speeds) > 1 else 0.0,
        "min": min(speeds),
        "max": max(speeds),
    }

    if verbose:
        print("\n" + "-" * 60)
        print(f"Average:  {stats['mean']:.2f} tok/s")
        print(f"Median:   {stats['median']:.2f} tok/s")
        print(f"Std Dev:  {stats['stdev']:.2f} tok/s")
        print(f"Min:      {stats['min']:.2f} tok/s")
        print(f"Max:      {stats['max']:.2f} tok/s")
        print("=" * 60)

    return speeds, stats


def print_comparison(
    llama_cpp_stats: Optional[Dict[str, float]] = None,
    llama_server_stats: Optional[Dict[str, float]] = None,
    ollama_stats: Optional[Dict[str, float]] = None,
):
    """
    Print comparison between llama-cpp, llama-cpp-server, and Ollama benchmarks.

    Args:
        llama_cpp_stats: Statistics from llama-cpp benchmark (optional)
        llama_server_stats: Statistics from llama-cpp-server benchmark (optional)
        ollama_stats: Statistics from Ollama benchmark (optional)
    """
    # Build list of available results
    results = {}
    headers = []

    if llama_cpp_stats is not None:
        results["llama-cpp"] = llama_cpp_stats
        headers.append("llama-cpp")
    if llama_server_stats is not None:
        results["llama-server"] = llama_server_stats
        headers.append("llama-server")
    if ollama_stats is not None:
        results["Ollama"] = ollama_stats
        headers.append("Ollama")

    if len(results) < 2:
        print("\n⚠ Need at least 2 benchmark results to compare")
        return

    print("\n" + "=" * 80)
    print("=== COMPARISON ===")
    print("=" * 80)

    # Dynamic header based on available results
    header_line = f"\n{'Metric':<12}"
    for name in headers:
        header_line += f" {name:>15}"
    header_line += f" {'Winner':>15}"
    print(header_line)
    print("-" * 80)

    for metric in ["mean", "median", "min", "max"]:
        line = f"{metric.capitalize():<12}"

        # Collect values
        values = {}
        for name, stats in results.items():
            values[name] = stats[metric]
            line += f" {stats[metric]:>12.2f} t/s"

        # Determine winner
        winner = max(values, key=values.get)
        line += f" {winner:>15}"

        print(line)

    # Performance differences section
    if len(results) > 1:
        print("\n" + "-" * 80)
        print("Performance Differences:")
        print("-" * 80)

        # Use the first result as baseline
        baseline_name = headers[0]
        baseline_mean = results[baseline_name]["mean"]

        print(f"Baseline: {baseline_name} ({baseline_mean:.2f} tok/s)")
        print()

        for name, stats in results.items():
            if name == baseline_name:
                continue
            diff = (stats["mean"] - baseline_mean) / baseline_mean * 100
            print(f"{name:20s}: {diff:+.1f}%")

    # Determine overall winner
    all_means = {name: stats["mean"] for name, stats in results.items()}
    fastest = max(all_means, key=all_means.get)
    fastest_speed = all_means[fastest]

    print("\n" + "=" * 80)
    print(f"✓ {fastest} is the fastest with {fastest_speed:.2f} tok/s average")
    print("=" * 80)


def main():
    """
    Main benchmark execution.
    """
    try:
        print("=" * 80)
        print("llama-cpp-python vs llama-cpp-server vs Ollama Benchmark")
        print("=" * 80)

        # Step 1: Benchmark llama-cpp-python
        print("\n[1/4] Benchmarking llama-cpp-python...")
        try:
            _, llama_cpp_stats = benchmark_local_llama_cpp()
        except Exception as e:
            print(f"✗ llama-cpp benchmark failed: {e}")
            sys.exit(1)

        # Step 2: Benchmark llama-cpp-server
        print("\n[2/4] Starting and benchmarking llama-cpp-server...")
        try:
            # Start server
            start_llama_cpp_server(MODEL_CONFIG)

            # Benchmark
            _, llama_server_stats = benchmark_llama_cpp_server()

            # Stop server
            stop_llama_cpp_server()

        except Exception as e:
            print(f"✗ llama-cpp-server benchmark failed: {e}")
            import traceback

            traceback.print_exc()
            stop_llama_cpp_server(verbose=False)
            sys.exit(1)

        # Step 3: Start Ollama server
        # prepare_ollama_server(OLLAMA_MODEL_NAME)
        #
        # # Step 4: Benchmark Ollama
        # print("\n[4/4] Benchmarking Ollama...")
        # try:
        #     _, ollama_stats = benchmark_ollama()
        # except Exception as e:
        #     print(f"✗ Ollama benchmark failed: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     sys.exit(1)

        # Step 5: Print comparison
        print_comparison(
            llama_cpp_stats,
            llama_server_stats,
            # ollama_stats
        )

        print("\n✓ Benchmark complete!")
        print("\nNote: Ollama server is still running. Stop it manually if needed:")
        print("  pkill -9 ollama")

    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        stop_llama_cpp_server()


if __name__ == "__main__":
    main()
