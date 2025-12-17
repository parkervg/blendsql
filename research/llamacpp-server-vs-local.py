import time
import statistics
import subprocess
import sys
import signal
import os
from pathlib import Path
import requests

from llama_cpp import Llama
from transformers import AutoTokenizer

# ---------------------------------------------------------------
# MODEL_NAME_OR_PATH = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
# FILENAME = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
# TOKENIZER_PATH = "Qwen/Qwen3-4B-Instruct-2507"
# CHAT_FORMAT = "qwen"
# Qwen: 200 tok/s local, 186 tok/s server

MODEL_NAME_OR_PATH = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
FILENAME = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
TOKENIZER_PATH = "meta-llama/Llama-3.1-8B-Instruct"
CHAT_FORMAT = "llama-3"
# Llama: 142 tok/s local, 135 tok/s server

# ---------------------------------------------------------------

MODEL_PATH = f"./models/{FILENAME}"
LITELLM_MODEL_PATH = "hosted_vllm/model"
SERVER_URL = "http://localhost:8000/v1"

PROMPT = """You are a helpful assistant.
Explain the theory of relativity in simple terms, with examples.
"""

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

MAX_TOKENS = 512
RUNS = 10


def tokens_per_second(token_count: int, elapsed: float) -> float:
    return token_count / elapsed


def benchmark_local_llama_cpp():
    print("\n=== Local llama-cpp-python ===")

    llm = Llama(
        model_path=MODEL_PATH,
        **{
            "n_gpu_layers": -1,
            "n_ctx": 8000,
            "seed": 100,
            "n_threads": 6,
            "chat_format": CHAT_FORMAT,
        },
    )

    speeds = []

    for i in range(RUNS):
        start = time.perf_counter()
        output = llm(
            PROMPT,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )
        elapsed = time.perf_counter() - start

        text = output["choices"][0]["text"]
        token_count = len(tokenizer.encode(text))

        speed = tokens_per_second(token_count, elapsed)
        speeds.append(speed)

        print(f"Run {i+1}: {token_count} tokens in {elapsed:.2f}s → {speed:.2f} tok/s")

    print(f"Average: {statistics.mean(speeds):.2f} tok/s")


def start_llama_cpp_server():
    print("\nStarting llama-cpp server...")

    cmd = [
        sys.executable,
        "-m",
        "llama_cpp.server",
        "--model",
        MODEL_PATH,
        "--n_ctx",
        "8000",
        "--seed",
        "100",
        "--n_threads",
        "6",
        "--n_gpu_layers",
        "-1",
        "--port",
        "8000",
        "--chat_format",
        CHAT_FORMAT,
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
    )

    # Stream output in background threads
    def stream_output(pipe, prefix):
        for line in iter(pipe.readline, ""):
            if line:
                print(f"{prefix}: {line.rstrip()}")
        pipe.close()

    stdout_thread = threading.Thread(
        target=stream_output, args=(process.stdout, "SERVER")
    )
    stderr_thread = threading.Thread(
        target=stream_output, args=(process.stderr, "SERVER")
    )
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()

    # Wait for server to be ready
    print("Waiting for server to be ready...")
    max_wait = 60  # Maximum wait time in seconds
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            response = requests.get("http://localhost:8000/v1/models", timeout=1)
            if response.status_code == 200:
                print("✓ Server is ready!")
                return process
        except (requests.ConnectionError, requests.Timeout):
            pass

        # Check if process has died
        if process.poll() is not None:
            raise RuntimeError("Server process terminated unexpectedly")

        time.sleep(0.5)

    # Timeout reached
    process.terminate()
    raise TimeoutError(f"Server failed to start within {max_wait} seconds")

    return process


def start_llama_cpp_server():
    print("\nStarting llama-cpp server...")

    cmd = [
        sys.executable,
        "-m",
        "llama_cpp.server",
        "--model",
        MODEL_PATH,
        "--n_ctx",
        "8000",
        "--seed",
        "100",
        "--n_threads",
        "6",
        "--n_gpu_layers",
        "-1",
        "--port",
        "8000",
        "--chat_format",
        CHAT_FORMAT,
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    wait_for_server(6)
    return process


def stop_server(process):
    print("\nStopping llama-cpp server...")
    process.send_signal(signal.SIGINT)
    process.wait(timeout=10)


if __name__ == "__main__":
    if not Path(MODEL_PATH).is_file():
        print("Downloading model...")
        os.system(
            f"""
        hf download \
        {MODEL_NAME_OR_PATH} \
        {FILENAME} \
        --local-dir ./models 
        """
        )
    benchmark_local_llama_cpp()

    server_process = start_llama_cpp_server()
    try:
        benchmark_hosted_via_litellm()
    finally:
        stop_server(server_process)
