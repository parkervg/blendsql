"""
Configuration settings for model evaluation framework.
"""
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

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
    "llama_3b": ModelConfig(
        model_name_or_path="unsloth/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q3_K_M.gguf",
        chat_format="llama-3",
    ),
    "llama_8b": ModelConfig(
        model_name_or_path="unsloth/Llama-3.1-8B-Instruct-GGUF",
        filename="Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        chat_format="llama-3",
    ),
    "smollm_3b": ModelConfig(
        model_name_or_path="unsloth/SmolLM3-3B-GGUF",
        filename="SmolLM3-3B-Q4_K_M.gguf",
        chat_format="chatml",
    ),
    "qwen_3b": ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        chat_format="qwen",
    ),
    "qwen_7b": ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct-GGUF",
        filename=[
            "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
            "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
        ],
        chat_format="qwen",
    ),
    "qwen_14b": ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-14B-Instruct-GGUF",
        filename=[
            "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
            "qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf",
            "qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf",
        ],
        chat_format="qwen",
    ),
}

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_ENDPOINT = f"{OLLAMA_BASE_URL}/api/tags"

current_date = datetime.now().strftime("%Y-%m-%d")
BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / current_date

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
SKIP_QUERIES = {"Q9", "Q10", "Q11", "Q12"}
ONLY_USE = {}

# Server Configuration
OLLAMA_SERVER_STARTUP_TIMEOUT = 10  # seconds
OLLAMA_SERVER_SHUTDOWN_DELAY = 2  # seconds
OLLAMA_WARMUP_TIMEOUT = 30  # seconds

# Flock Configuration
FLOCK_VERSION = "7f1c36a"
