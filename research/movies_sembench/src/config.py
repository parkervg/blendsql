"""
Configuration settings for model evaluation framework.
"""
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
USE_DATA_SIZE = 2000
DUCKDB_SEED = 0.5
BASE_URL = "http://localhost:8000/v1"

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


@dataclass
class ModelConfig:
    model_name_or_path: str


MODEL_CONFIGS = {
    "gemma_12b": ModelConfig(
        model_name_or_path="RedHatAI/gemma-3-12b-it-quantized.w4a16",
    ),
    "gemma_4b": ModelConfig(
        model_name_or_path="RedHatAI/gemma-3-4b-it-quantized.w4a16",
    ),
    "qwen_4b": ModelConfig(
        model_name_or_path="Qwen/Qwen3-4B-Instruct-2507-FP8",
    ),
    "qwen_7b": ModelConfig(
        model_name_or_path="RedHatAI/Qwen2.5-VL-7B-Instruct-quantized.w4a16",
    ),
}

current_date = datetime.now().strftime("%Y-%m-%d")
BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / current_date

# System params
N_PARALLEL = 32

# Paths
MOVIE_FILES_DIR = BASE_DIR / "data"
DUCKDB_DB_PATH = MOVIE_FILES_DIR / f"movie_database_{USE_DATA_SIZE}.duckdb"
QUERIES_DIR = BASE_DIR / "queries"
THALAMUS_CONFIG_PATH = "../thalamus_db_model_config.json"

# Query Filtering
SKIP_QUERIES = {"Q9", "Q10", "Q11", "Q12"}
ONLY_USE = {}

# Server Configuration
OLLAMA_SERVER_STARTUP_TIMEOUT = 10  # seconds
OLLAMA_SERVER_SHUTDOWN_DELAY = 2  # seconds
OLLAMA_WARMUP_TIMEOUT = 30  # seconds

# Flock Configuration
FLOCK_VERSION = "7f1c36a"
