"""
Configuration settings for model evaluation framework.
"""
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent
USE_DATA_SIZE = 2000
DUCKDB_SEED = 0.5


@dataclass
class ModelConfig:
    model_name_or_path: str
    filename: str
    chat_format: str


MODEL_CONFIGS = {
    "gemma_4b": ModelConfig(
        model_name_or_path="unsloth/gemma-3-4b-it-GGUF",
        filename="gemma-3-4b-it-Q4_K_M.gguf",
        chat_format="gemma",
    ),
    "gemma_12b": ModelConfig(
        model_name_or_path="unsloth/gemma-3-12b-it-GGUF",
        filename="gemma-3-12b-it-Q4_K_M.gguf",
        chat_format="gemma",
    ),
    "qwen_4b": ModelConfig(
        model_name_or_path="Qwen/Qwen3-4B-GGUF",
        filename="Qwen3-4B-Q4_K_M.gguf",
        chat_format="qwen",
    ),
    "qwen_14b": ModelConfig(
        model_name_or_path="Qwen/Qwen3-14B-GGUF",
        filename="Qwen3-14B-Q4_K_M.gguf",
        chat_format="qwen",
    ),
}

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_ENDPOINT = f"{OLLAMA_BASE_URL}/api/tags"

BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"

# Model Parameters
MODEL_PARAMS = {
    "temperature": 0.0,
    "repeat_penalty": 1.0,
    "max_tokens": 5,
    "num_ctx": 2048,
    "seed": 100,
    "num_predict": -1,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "num_threads": 6,
    "n_gpu_layers": -1,
    "flash_attn": True,
    "n_batch": 2048,
}

# System params
SYSTEM_PARAMS = {"batch_size": 1}

# Paths
MOVIE_FILES_DIR = BASE_DIR / "data"
DUCKDB_DB_PATH = MOVIE_FILES_DIR / f"movie_database_{USE_DATA_SIZE}.duckdb"
QUERIES_DIR = BASE_DIR / "queries"
THALAMUS_CONFIG_PATH = "../thalamus_db_model_config.json"

# Query Filtering
SKIP_QUERIES = {"Q7"}
ONLY_USE = {}

# Server Configuration
OLLAMA_SERVER_STARTUP_TIMEOUT = 10  # seconds
OLLAMA_SERVER_SHUTDOWN_DELAY = 2  # seconds
OLLAMA_WARMUP_TIMEOUT = 30  # seconds

# Flock Configuration
FLOCK_VERSION = "7f1c36a"
