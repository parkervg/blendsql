"""
Configuration settings for model evaluation framework.
"""
from pathlib import Path
import os

# Model Configuration
MODEL_NAME_OR_PATH = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
FILENAME = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

# Ollama Configuration
MODEL_NAME = Path(FILENAME).stem
LOCAL_GGUF_FILEPATH = Path(__file__).resolve().parent / f"models/{FILENAME}"
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
}

# System params
SYSTEM_PARAMS = {"batch_size": 10}

# Paths
BASE_DIR = Path(os.path.abspath(Path(__file__).resolve().parent))
MOVIE_FILES_DIR = BASE_DIR / "data"
DUCKDB_DB_PATH = MOVIE_FILES_DIR / "movie_database.duckdb"
QUERIES_DIR = BASE_DIR / "queries"
THALAMUS_CONFIG_PATH = "../thalamus_db_model_config.json"

# Evaluation Configuration
EVALS_TO_RUN = {
    "blendsql": True,
    "flock": False,
    "thalamusdb": True,
}

# Query Filtering
SKIP_QUERIES = {"Q10"}
ONLY_USE = {}

# Server Configuration
OLLAMA_SERVER_STARTUP_TIMEOUT = 10  # seconds
OLLAMA_SERVER_SHUTDOWN_DELAY = 2  # seconds
OLLAMA_WARMUP_TIMEOUT = 30  # seconds

# Flock Configuration
FLOCK_VERSION = "7f1c36a"
FLOCK_BATCH_SIZE = 32
FLOCK_TEMPERATURE = 0.7

# Template for Ollama Modelfile
OLLAMA_MODELFILE_TEMPLATE = """FROM {gguf_path}

# Set the template format
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
\"\"\"

# Set parameters 
# https://docs.ollama.com/modelfile#instructions
# Match the llama-cpp-python defaults
# https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#__codelineno-0-670
PARAMETER temperature {temperature}
PARAMETER repeat_penalty {repeat_penalty}
PARAMETER num_ctx {num_ctx}
PARAMETER seed {seed}
PARAMETER num_predict {num_predict}
PARAMETER top_k {top_k}
PARAMETER top_p {top_p}
PARAMETER min_p {min_p}
PARAMETER num_thread {num_thread}
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
"""
