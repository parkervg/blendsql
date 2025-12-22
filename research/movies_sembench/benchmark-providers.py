import sys
import gc

from src.ollama_utils import (
    prepare_ollama_server,
    stop_ollama_server,
)
from src.config import (
    OLLAMA_BASE_URL,
    MODEL_PARAMS,
)

import time
import statistics
from typing import List, Dict, Tuple

from llama_cpp import Llama
from transformers import AutoTokenizer
import requests

BENCHMARK_PROMPT = """You are a helpful assistant.
Explain the theory of relativity in simple terms, with LOTS of examples.
"""
BENCHMARK_RUNS = 20
BENCHMARK_TEMPERATURE = 0.7
MAX_TOKENS = 512

N_CTX = MODEL_PARAMS["num_ctx"]
SEED = MODEL_PARAMS["seed"]
N_THREADS = MODEL_PARAMS["num_threads"]
CHAT_FORMAT = "gemma3"

MODEL_NAME_OR_PATH = "unsloth/gemma-3-4b-it-GGUF"
FILENAME = "gemma-3-4b-it-Q4_K_M.gguf"
OLLAMA_MODEL_NAME = "gemma3:4b"
TOKENIZER_PATH = "google/gemma-3-4b-it"

OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/v1"
OLLAMA_GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"

BENCHMARK_LLAMA_CPP_CONFIG = {
    "n_gpu_layers": -1,
    "n_ctx": N_CTX,
    "seed": SEED,
    "n_threads": N_THREADS,
    "chat_format": CHAT_FORMAT,
    "flash_attn": MODEL_PARAMS["flash_attn"],
    # "flash_attn": False # Is ollama using flash_attn correctly?
}


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
        repo_id=str(MODEL_NAME_OR_PATH),
        filename=str(FILENAME),
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


def benchmark_ollama(
    prompt: str = BENCHMARK_PROMPT,
    max_tokens: int = MAX_TOKENS,
    runs: int = BENCHMARK_RUNS,
    temperature: float = BENCHMARK_TEMPERATURE,
    verbose: bool = True,
    use_openai_api: bool = False,
) -> Tuple[List[float], Dict[str, float]]:
    """
    Benchmark Ollama inference.

    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        runs: Number of benchmark runs
        temperature: Sampling temperature
        verbose: Print progress
        use_openai_api: Use OpenAI-compatible API instead of native Ollama API

    Returns:
        Tuple of (speeds list, stats dict)
    """
    if verbose:
        print("\n" + "=" * 60)
        api_type = "OpenAI-compatible" if use_openai_api else "Native"
        print(f"=== Benchmarking Ollama ({api_type} API) ===")
        print("=" * 60)

    # Load tokenizer for counting
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    if verbose:
        print(f"Model: {OLLAMA_MODEL_NAME}")
        print(f"\nRunning {runs} iterations...")
        print(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")
        print(f"Max tokens: {max_tokens}")
        print(f"Temperature: {temperature}\n")

    speeds = []

    # Use native Ollama API
    for i in range(runs):
        start = time.perf_counter()

        response = requests.post(
            OLLAMA_GENERATE_ENDPOINT,
            json={
                "model": OLLAMA_MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                },
            },
        )

        elapsed = time.perf_counter() - start

        # Get generated text and count tokens
        result = response.json()
        text = result.get("response", "")
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
    llama_cpp_stats: Dict[str, float],
    ollama_stats: Dict[str, float],
):
    """
    Print comparison between llama-cpp and Ollama benchmarks.

    Args:
        llama_cpp_stats: Statistics from llama-cpp benchmark
        ollama_stats: Statistics from Ollama benchmark
    """
    print("\n" + "=" * 60)
    print("=== COMPARISON ===")
    print("=" * 60)

    print(f"\n{'Metric':<15} {'llama-cpp':>15} {'Ollama':>15} {'Difference':>15}")
    print("-" * 60)

    for metric in ["mean", "median", "min", "max"]:
        llama_val = llama_cpp_stats[metric]
        ollama_val = ollama_stats[metric]
        diff = llama_val - ollama_val
        diff_pct = (diff / ollama_val * 100) if ollama_val != 0 else 0

        print(
            f"{metric.capitalize():<15} {llama_val:>12.2f} t/s {ollama_val:>12.2f} t/s {diff:>+10.2f} ({diff_pct:+.1f}%)"
        )

    print("=" * 60)

    # Determine winner
    if llama_cpp_stats["mean"] > ollama_stats["mean"]:
        diff_pct = (
            (llama_cpp_stats["mean"] - ollama_stats["mean"])
            / ollama_stats["mean"]
            * 100
        )
        print(f"\n✓ llama-cpp is {diff_pct:.1f}% faster on average")
    elif ollama_stats["mean"] > llama_cpp_stats["mean"]:
        diff_pct = (
            (ollama_stats["mean"] - llama_cpp_stats["mean"])
            / llama_cpp_stats["mean"]
            * 100
        )
        print(f"\n✓ Ollama is {diff_pct:.1f}% faster on average")
    else:
        print("\n= Performance is equivalent")

    print("=" * 60)


def main():
    """
    Main benchmark execution.
    """
    try:
        print("=" * 60)
        print("llama-cpp-python vs Ollama Benchmark")
        print("=" * 60)

        # Step 2: Benchmark llama-cpp-python
        print("\n[2/5] Benchmarking llama-cpp-python...")
        try:
            _, llama_cpp_stats = benchmark_local_llama_cpp()
        except Exception as e:
            print(f"✗ llama-cpp benchmark failed: {e}")
            sys.exit(1)

        # Step 3: Start Ollama server
        prepare_ollama_server(OLLAMA_MODEL_NAME)

        # Step 4: Benchmark Ollama
        print("\n[5/5] Benchmarking Ollama...")
        try:
            # Benchmark using native Ollama API
            _, ollama_stats = benchmark_ollama()
        except Exception as e:
            print(f"✗ Ollama benchmark failed: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        # Step 5: Print comparison
        print_comparison(llama_cpp_stats, ollama_stats)

        print("\n✓ Benchmark complete!")
        print("\nNote: Ollama server is still running. Stop it manually if needed:")
        print("  pkill -9 ollama")
    except:
        raise
    finally:
        stop_ollama_server()


if __name__ == "__main__":
    main()
