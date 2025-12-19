import subprocess
import time
import requests
from pathlib import Path
import sys

from blendsql.common.logger import Color, logger

from src.config import (
    MODEL_PARAMS,
    OLLAMA_API_ENDPOINT,
    OLLAMA_BASE_URL,
    OLLAMA_SERVER_STARTUP_TIMEOUT,
    OLLAMA_SERVER_SHUTDOWN_DELAY,
    OLLAMA_WARMUP_TIMEOUT,
    OLLAMA_MODELFILE_TEMPLATE,
)


def check_ollama_running() -> bool:
    """
    Check if Ollama server is running.

    Returns:
        True if server is running, False otherwise
    """
    try:
        response = requests.get(OLLAMA_API_ENDPOINT, timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def start_ollama_server() -> bool:
    """
    Start Ollama server if not running.

    Returns:
        True if server is running (was already running or started successfully)
    """
    if check_ollama_running():
        logger.debug(Color.success("✓ Ollama server is already running"))
        return True

    logger.debug(Color.update("Starting Ollama server..."))
    try:
        # Start Ollama in the background
        subprocess.Popen(
            ["OLLAMA_FLASH_ATTENTION=1", "ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for server to start
        for _i in range(OLLAMA_SERVER_STARTUP_TIMEOUT):
            time.sleep(1)
            if check_ollama_running():
                logger.debug(Color.success("✓ Ollama server started successfully"))
                return True

        logger.debug(
            Color.warning("Warning: Ollama server may not have started properly")
        )
        return False

    except FileNotFoundError:
        print("Error: 'ollama' command not found. Make sure Ollama is installed.")
        return False


def stop_ollama_server() -> bool:
    """
    Stop Ollama server.

    Returns:
        True if server stopped successfully
    """
    logger.debug(Color.update("Stopping Ollama server..."))

    try:
        subprocess.run(["pkill", "-9", "ollama"], capture_output=True, check=False)

        # Wait a moment and verify
        time.sleep(OLLAMA_SERVER_SHUTDOWN_DELAY)
        if not check_ollama_running():
            logger.debug(Color.success("✓ Ollama server stopped successfully"))
            return True
        else:
            logger.debug(Color.warning("Warning: Ollama server may still be running"))
            return False

    except Exception as e:
        logger.debug(Color.error(f"Error stopping Ollama: {e}"))
        return False


def create_ollama_modelfile(gguf_path: str) -> Path:
    """
    Create a Modelfile for Ollama.

    Args:
        gguf_path: Path to the GGUF model file

    Returns:
        Path to the created Modelfile
    """
    modelfile_path = Path("./Modelfile")

    modelfile_content = OLLAMA_MODELFILE_TEMPLATE.format(
        gguf_path=gguf_path,
        temperature=MODEL_PARAMS["temperature"],
        repeat_penalty=MODEL_PARAMS["repeat_penalty"],
        num_ctx=MODEL_PARAMS["num_ctx"],
        seed=MODEL_PARAMS["seed"],
        num_predict=MODEL_PARAMS["num_predict"],
        top_k=MODEL_PARAMS["top_k"],
        top_p=MODEL_PARAMS["top_p"],
        min_p=MODEL_PARAMS["min_p"],
        num_thread=MODEL_PARAMS["num_threads"],
    )

    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    return modelfile_path


def load_gguf_into_ollama(gguf_path: str, model_name: str) -> bool:
    """
    Load a GGUF file into Ollama by creating a model.

    Args:
        gguf_path: Path to the GGUF model file
        model_name: Name to give the model in Ollama

    Returns:
        True if model loaded successfully
    """
    logger.debug(Color.update(f"Loading GGUF model into Ollama as '{model_name}'..."))

    # Create Modelfile
    modelfile_path = create_ollama_modelfile(gguf_path)

    try:
        # Create the model in Ollama
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.debug(
            Color.success(f"✓ Model '{model_name}' created successfully in Ollama")
        )
        logger.debug(Color.update(result.stdout))

        # Clean up Modelfile
        modelfile_path.unlink()
        return True

    except subprocess.CalledProcessError as e:
        logger.debug(Color.error(f"Error creating Ollama model: {e}"))
        logger.debug(Color.error(f"stdout: {e.stdout}"))
        logger.debug(Color.error(f"stderr: {e.stderr}"))
        return False
    except FileNotFoundError:
        logger.debug(
            Color.error(
                "Error: 'ollama' command not found. Make sure Ollama is installed and in your PATH"
            )
        )
        return False


def warmup_ollama_model(model_name: str, base_url: str = OLLAMA_BASE_URL) -> bool:
    """
    Warm up Ollama model by making a simple request.

    Args:
        model_name: Name of the model to warm up
        base_url: Base URL of Ollama server

    Returns:
        True if warmup successful
    """
    logger.debug(Color.update(f"Warming up Ollama model: {model_name}..."))

    try:
        # Use Ollama's native API for warmup (faster than OpenAI endpoint)
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": "Hi",
                "stream": False,
                "options": {"num_predict": 1},  # Generate just 1 token
            },
            timeout=OLLAMA_WARMUP_TIMEOUT,
        )

        if response.status_code == 200:
            logger.debug(Color.success("✓ Model warmed up successfully!"))
            return True
        else:
            logger.debug(
                Color.warning(f"Warning: Warmup returned status {response.status_code}")
            )
            return False

    except requests.exceptions.RequestException as e:
        logger.debug(Color.error(f"Warning: Model warmup failed: {e}"))
        return False


def prepare_ollama_server(gguf_filepath: str, model_name: str):
    # Start Ollama server
    start_ollama_server()

    # Load GGUF into Ollama
    if not load_gguf_into_ollama(gguf_filepath, model_name):
        logger.debug(Color.error("Failed to load model into Ollama. Exiting."))
        sys.exit(1)

    # Warm up the model
    warmup_ollama_model(model_name)
