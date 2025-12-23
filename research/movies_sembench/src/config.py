"""
Configuration settings for model evaluation framework.
"""
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent
USE_DATA_SIZE = 2000
MODEL_SIZE = "4b"


@dataclass
class ModelConfig:
    model_name_or_path: str
    filename: str
    ollama_model_name: str


MODEL_CONFIGS = {
    "4b": ModelConfig(
        model_name_or_path="unsloth/gemma-3-4b-it-GGUF",
        filename="gemma-3-4b-it-Q4_K_M.gguf",
        ollama_model_name="gemma3:4b",
    ),
    "12b": ModelConfig(
        model_name_or_path="unsloth/gemma-3-12b-it-GGUF",
        filename="gemma-3-12b-it-Q4_K_M.gguf",
        ollama_model_name="gemma3:12b",
    ),
}

MODEL_CONFIG = MODEL_CONFIGS[MODEL_SIZE]

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_ENDPOINT = f"{OLLAMA_BASE_URL}/api/tags"

OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# Model Parameters
MODEL_PARAMS = {
    "temperature": 0.0,
    "repeat_penalty": 1.0,
    "num_ctx": 2048,
    "seed": 100,
    "num_predict": -1,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "num_threads": 6,
    "n_gpu_layers": -1,
    "flash_attn": True,
    "n_batch": 512,
}

# System params
SYSTEM_PARAMS = {"batch_size": 5}

# Paths
MOVIE_FILES_DIR = BASE_DIR / "data"
DUCKDB_DB_PATH = MOVIE_FILES_DIR / f"movie_database_{USE_DATA_SIZE}.duckdb"
QUERIES_DIR = BASE_DIR / "queries"
THALAMUS_CONFIG_PATH = "../thalamus_db_model_config.json"

# Query Filtering
SKIP_QUERIES = {"Q10", "Q7"}
ONLY_USE = {}

# Server Configuration
OLLAMA_SERVER_STARTUP_TIMEOUT = 10  # seconds
OLLAMA_SERVER_SHUTDOWN_DELAY = 2  # seconds
OLLAMA_WARMUP_TIMEOUT = 30  # seconds

# Flock Configuration
FLOCK_VERSION = "7f1c36a"
