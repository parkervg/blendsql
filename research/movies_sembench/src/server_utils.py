import subprocess
import signal
import os
import time

from .config import ModelConfig

# Global variable to store the server process
_vllm_process = None


def start_vllm_server(
    model_config: ModelConfig,
    host: str = "0.0.0.0",
    port: int = 8000,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.8,
    enable_prefix_caching: bool = True,
    enable_prompt_tokens_details: bool = True,
    structured_outputs_backend: str = "guidance",
    wait_for_ready: bool = True,
    timeout: int = 300,
) -> subprocess.Popen:
    """
    Start a vLLM server in the background.

    Args:
        model_name_or_path: The model to serve
        host: Host address to bind to
        port: Port to listen on
        max_model_len: Maximum model context length
        gpu_memory_utilization: Fraction of GPU memory to use
        enable_prefix_caching: Enable prefix caching optimization
        enable_prompt_tokens_details: Enable prompt token details in response
        structured_outputs_backend: Backend for structured outputs
        wait_for_ready: Wait for server to be ready before returning
        timeout: Timeout in seconds when waiting for server

    Returns:
        The subprocess.Popen object for the server process
    """
    global _vllm_process

    if _vllm_process is not None and _vllm_process.poll() is None:
        print("vLLM server is already running")
        return _vllm_process

    cmd = [
        "vllm",
        "serve",
        model_config.model_name_or_path,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
        "--structured-outputs-config.backend",
        structured_outputs_backend,
    ]

    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    if enable_prompt_tokens_details:
        cmd.append("--enable-prompt-tokens-details")

    print(f"Starting vLLM server with command: {' '.join(cmd)}")

    _vllm_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # Create new process group for clean shutdown
    )

    if wait_for_ready:
        import urllib.request
        import urllib.error

        health_url = (
            f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}/health"
        )
        start_time = time.time()

        print(f"Waiting for server to be ready at {health_url}...")
        while time.time() - start_time < timeout:
            # Check if process died
            if _vllm_process.poll() is not None:
                stdout, _ = _vllm_process.communicate()
                raise RuntimeError(
                    f"vLLM server process died. Output:\n{stdout.decode()}"
                )

            try:
                with urllib.request.urlopen(health_url, timeout=5) as response:
                    if response.status == 200:
                        print("vLLM server is ready!")
                        return _vllm_process
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                ConnectionRefusedError,
            ):
                pass

            time.sleep(2)

        raise TimeoutError(f"vLLM server did not become ready within {timeout} seconds")

    return _vllm_process


def stop_vllm_server(timeout: int = 30) -> bool:
    """
    Stop the vLLM server that was started by start_vllm_server.

    Args:
        timeout: Timeout in seconds to wait for graceful shutdown

    Returns:
        True if server was stopped, False if no server was running
    """
    global _vllm_process

    if _vllm_process is None:
        print("No vLLM server process to stop")
        return False

    if _vllm_process.poll() is not None:
        print("vLLM server process already terminated")
        _vllm_process = None
        return False

    print("Stopping vLLM server...")

    # Send SIGTERM to the process group for graceful shutdown
    try:
        os.killpg(os.getpgid(_vllm_process.pid), signal.SIGTERM)
    except ProcessLookupError:
        print("Process already terminated")
        _vllm_process = None
        return True

    # Wait for graceful shutdown
    try:
        _vllm_process.wait(timeout=timeout)
        print("vLLM server stopped gracefully")
    except subprocess.TimeoutExpired:
        print(f"Server did not stop within {timeout}s, sending SIGKILL...")
        try:
            os.killpg(os.getpgid(_vllm_process.pid), signal.SIGKILL)
            _vllm_process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        print("vLLM server killed")

    _vllm_process = None
    return True


# Example usage
if __name__ == "__main__":
    try:
        # Start the server
        process = start_vllm_server(wait_for_ready=True)
        print(f"Server running with PID: {process.pid}")

        # Keep it running for a bit (or do your work here)
        time.sleep(10)

    finally:
        # Always stop the server
        stop_vllm_server()
