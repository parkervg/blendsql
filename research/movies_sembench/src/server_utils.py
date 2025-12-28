import subprocess
import atexit
import time
import requests
from huggingface_hub import hf_hub_download

from .config import MODEL_PARAMS, ModelConfig

LLAMA_SERVER_HOST = "127.0.0.1"
LLAMA_SERVER_PORT = 8080
LLAMA_SERVER_BASE_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
LLAMA_SERVER_COMPLETION_ENDPOINT = f"{LLAMA_SERVER_BASE_URL}/v1/completions"


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


def start_llama_cpp_server(
    model_config: ModelConfig,
    host: str = LLAMA_SERVER_HOST,
    port: int = LLAMA_SERVER_PORT,
    n_gpu_layers: int = -1,
    verbose=False,
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

    model_path = hf_hub_download(
        repo_id=model_config.model_name_or_path,
        filename=model_config.filename,
    )

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
        str(MODEL_PARAMS["num_ctx"]),
        "--n_threads",
        str(MODEL_PARAMS["num_threads"]),
        "--flash_attn",
        str(MODEL_PARAMS["flash_attn"]).lower(),
        "--seed",
        str(MODEL_PARAMS["seed"]),
        "--n_batch",
        str(MODEL_PARAMS["n_batch"]),
        "--chat_format",
        model_config.chat_format,
    ]

    # Start server
    _llama_server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Register cleanup on exit
    atexit.register(stop_llama_cpp_server, verbose=verbose)

    # Wait for server to be ready
    max_wait = 15  # seconds
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            # Try the models endpoint which should be available when server is ready
            response = requests.get(f"{LLAMA_SERVER_BASE_URL}/v1/models", timeout=1)
            if response.status_code == 200:
                if verbose:
                    print("✓ llama-cpp-server is ready")
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    else:
        raise RuntimeError("llama-cpp-server failed to start within timeout period")

    # Warmup call to completions endpoint
    if verbose:
        print("Running warmup call...")

    try:
        warmup_response = requests.post(
            f"{LLAMA_SERVER_BASE_URL}/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "temperature": 0.0,
            },
            timeout=30,
        )
        if warmup_response.status_code == 200:
            if verbose:
                print("✓ Warmup call completed successfully")
        else:
            if verbose:
                print(f"⚠ Warmup call returned status {warmup_response.status_code}")
    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"⚠ Warmup call failed: {e}")

    return _llama_server_process
