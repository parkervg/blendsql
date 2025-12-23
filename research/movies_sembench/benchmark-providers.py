import sys
import gc
import subprocess
import atexit

from src.ollama_utils import (
    stop_ollama_server,
)
from src.config import (
    OLLAMA_BASE_URL,
    MODEL_PARAMS,
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

N_CTX = MODEL_PARAMS["num_ctx"]
SEED = MODEL_PARAMS["seed"]
N_THREADS = MODEL_PARAMS["num_threads"]
FLASH_ATTN = MODEL_PARAMS["flash_attn"]
N_BATCH = MODEL_PARAMS["n_batch"]
CHAT_FORMAT = "gemma3"

MODEL_NAME_OR_PATH = "unsloth/gemma-3-4b-it-GGUF"
FILENAME = "gemma-3-4b-it-Q4_K_M.gguf"
OLLAMA_MODEL_NAME = "gemma3:4b"
TOKENIZER_PATH = "google/gemma-3-4b-it"

OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/v1"
OLLAMA_GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"

# llama-cpp-server configuration
LLAMA_SERVER_HOST = "127.0.0.1"
LLAMA_SERVER_PORT = 8080
LLAMA_SERVER_BASE_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
LLAMA_SERVER_COMPLETION_ENDPOINT = f"{LLAMA_SERVER_BASE_URL}/v1/completions"

BENCHMARK_LLAMA_CPP_CONFIG = {
    "n_gpu_layers": -1,
    "n_ctx": N_CTX,
    "seed": SEED,
    "n_threads": N_THREADS,
    "chat_format": CHAT_FORMAT,
    "flash_attn": FLASH_ATTN,
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


def start_llama_cpp_server(
    model_path: str,
    host: str = LLAMA_SERVER_HOST,
    port: int = LLAMA_SERVER_PORT,
    n_gpu_layers: int = -1,
    n_ctx: int = N_CTX,
    n_threads: int = N_THREADS,
    flash_attn: bool = FLASH_ATTN,
    seed: int = SEED,
    n_batch: int = N_BATCH,
    verbose: bool = True,
) -> subprocess.Popen:
    """
    Start llama-cpp-server.

    Args:
        model_path: Path to the GGUF model file
        host: Server host
        port: Server port
        n_gpu_layers: Number of layers to offload to GPU (-1 for all)
        n_ctx: Context size
        n_threads: Number of threads
        flash_attn: Enable flash attention
        verbose: Print server output

    Returns:
        Server process
    """
    global _llama_server_process

    if verbose:
        print(f"\nStarting llama-cpp-server on {host}:{port}...")

    # Build command
    cmd = [
        "python",
        "-m",
        "llama_cpp.server",
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--n_gpu_layers",
        str(n_gpu_layers),
        "--n_ctx",
        str(n_ctx),
        "--n_threads",
        str(n_threads),
        "--flash_attn",
        str(flash_attn).lower(),
        "--seed",
        str(seed),
        "--n_batch",
        str(n_batch),
    ]

    # Start server
    _llama_server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if not verbose else None,
        stderr=subprocess.PIPE if not verbose else None,
    )

    # Register cleanup on exit
    atexit.register(stop_llama_cpp_server, verbose=verbose)

    # Wait for server to be ready
    max_wait = 60  # seconds
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            # Try the models endpoint which should be available when server is ready
            response = requests.get(f"{LLAMA_SERVER_BASE_URL}/v1/models", timeout=1)
            if response.status_code == 200:
                if verbose:
                    print("✓ llama-cpp-server is ready")
                return _llama_server_process
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    raise RuntimeError("llama-cpp-server failed to start within timeout period")


def stop_llama_cpp_server(verbose: bool = True):
    """
    Stop llama-cpp-server.

    Args:
        verbose: Print status messages
    """
    global _llama_server_process

    if _llama_server_process is None:
        return

    if verbose:
        print("\nStopping llama-cpp-server...")

    try:
        # Try graceful shutdown first
        _llama_server_process.terminate()
        try:
            _llama_server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if graceful shutdown fails
            _llama_server_process.kill()
            _llama_server_process.wait()

        if verbose:
            print("✓ llama-cpp-server stopped")
    except Exception as e:
        if verbose:
            print(f"⚠ Error stopping server: {e}")
    finally:
        _llama_server_process = None


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


def benchmark_ollama(
    prompt: str = BENCHMARK_PROMPT,
    max_tokens: int = MAX_TOKENS,
    runs: int = BENCHMARK_RUNS,
    temperature: float = BENCHMARK_TEMPERATURE,
    verbose: bool = True,
) -> Tuple[List[float], Dict[str, float]]:
    """
    Benchmark Ollama inference.

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
        api_type = "Native"
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
            # Download model to cache if needed
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(
                repo_id=MODEL_NAME_OR_PATH,
                filename=FILENAME,
            )

            # Start server
            start_llama_cpp_server(
                model_path=model_path,
                n_gpu_layers=BENCHMARK_LLAMA_CPP_CONFIG["n_gpu_layers"],
                n_ctx=BENCHMARK_LLAMA_CPP_CONFIG["n_ctx"],
                n_threads=BENCHMARK_LLAMA_CPP_CONFIG["n_threads"],
                flash_attn=BENCHMARK_LLAMA_CPP_CONFIG["flash_attn"],
            )

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
        stop_llama_cpp_server(verbose=False)
        stop_ollama_server()


if __name__ == "__main__":
    main()
