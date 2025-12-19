import os
from pathlib import Path
import subprocess
import time
import requests
from textwrap import dedent
import logging
from typing import Generator

import pandas as pd
import duckdb
import sys

from blendsql.common.logger import Color, logger
from blendsql.common.utils import fetch_from_hub

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logger.setLevel(logging.DEBUG)

EVALS_TO_RUN = {"blendsql": True, "flock": False, "thalamusdb": True}

# MODEL_NAME_OR_PATH="unsloth/Qwen3-4B-Instruct-2507-GGUF"
# FILENAME="Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

SKIP_QUERIES = {}
ONLY_USE = {"Q1"}

# Ollama model name (can be different from filename)
MODEL_NAME = "qwen2.5-1.5b-custom"
LOCAL_GGUF_FILEPATH = f"./models/{FILENAME}"

MOVIE_FILES_DIR = Path(os.path.abspath(Path(__file__).resolve().parent / "data"))
DUCKDB_DB_PATH = MOVIE_FILES_DIR / "movie_database.duckdb"


def create_ollama_modelfile(gguf_path):
    """Create a Modelfile for Ollama"""
    modelfile_path = Path("./Modelfile")

    modelfile_content = f"FROM {gguf_path}" + dedent(
        """ 

    # Set the template format
    TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
    {{ .System }}<|im_end|>
    {{ end }}{{ if .Prompt }}<|im_start|>user
    {{ .Prompt }}<|im_end|>
    {{ end }}<|im_start|>assistant
    \"\"\"
    
    # Set parameters 
    # https://docs.ollama.com/modelfile#instructions
    # Match the llama-cpp-python defaults
    # https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#__codelineno-0-670
    PARAMETER temperature 0.7
    PARAMETER repeat_penalty 1.0
    PARAMETER num_ctx 2048
    PARAMETER seed 100
    PARAMETER num_predict -1
    PARAMETER top_k 40
    PARAMETER top_p 0.95
    PARAMETER min_p 0.05
    PARAMETER stop "<|im_start|>"
    PARAMETER stop "<|im_end|>"
    """
    )

    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    return modelfile_path


def load_gguf_into_ollama(gguf_path, model_name):
    """Load a GGUF file into Ollama by creating a model"""
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


def check_ollama_running():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def start_ollama_server():
    """Start Ollama server if not running"""
    if check_ollama_running():
        logger.debug(Color.success("✓ Ollama server is already running"))
        return True

    logger.debug(Color.update("Starting Ollama server..."))
    try:
        # Start Ollama in the background
        subprocess.Popen(
            ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # Wait for server to start
        for _i in range(10):
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


def stop_ollama_server():
    """Stop Ollama server"""
    logger.debug(Color.update("Stopping Ollama server..."))

    try:
        subprocess.run(["pkill", "-9", "ollama"], capture_output=True, check=False)

        # Wait a moment and verify
        time.sleep(2)
        if not check_ollama_running():
            logger.debug(Color.success("✓ Ollama server stopped successfully"))
            return True
        else:
            logger.debug(Color.warning("Warning: Ollama server may still be running"))
            return False

    except Exception as e:
        logger.debug(Color.error(f"Error stopping Ollama: {e}"))
        return False


def warmup_ollama_model(model_name, base_url="http://localhost:11434"):
    """Warm up Ollama model by making a simple request"""
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
            timeout=30,
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


def create_duckdb_database():
    """Not sure if other methods modify database state - so we want to reset before each eval"""
    logger.debug(Color.update("Creating database..."))
    if DUCKDB_DB_PATH.is_file():
        DUCKDB_DB_PATH.unlink()

    conn = duckdb.connect(DUCKDB_DB_PATH)
    movies_df = pd.read_csv(fetch_from_hub("movie/rotten_tomatoes_movies.csv"))
    reviews_df = pd.read_csv(fetch_from_hub("movie/rotten_tomatoes_movie_reviews.csv"))

    # Step 2: Save DataFrames as persistent tables
    conn.register("movies_df", movies_df)
    conn.register("reviews_df", reviews_df)

    conn.execute("CREATE OR REPLACE TABLE Movies AS SELECT * FROM movies_df")
    conn.execute("CREATE OR REPLACE TABLE Reviews AS SELECT * FROM reviews_df")
    logger.debug(Color.success("Database tables created successfully"))
    conn.close()


def iter_queries(system_name: str) -> Generator:
    for query_file in (
        Path(__file__).resolve().parent / f"queries/{system_name}"
    ).iterdir():
        query_name = query_file.stem
        if query_name in SKIP_QUERIES:
            continue
        if ONLY_USE and query_name not in ONLY_USE:
            continue
        logger.debug(Color.update(f"Running {system_name} {query_name}..."))
        yield (query_file, query_name)


if __name__ == "__main__":
    # Download GGUF if not exists
    if not Path(LOCAL_GGUF_FILEPATH).is_file():
        logger.debug(
            Color.update(f"Downloading model \n {MODEL_NAME_OR_PATH=} {FILENAME=}")
        )
        os.system(
            f"""
        hf download \
        {MODEL_NAME_OR_PATH} \
        {FILENAME} \
        --local-dir ./models 
        """
        )

    if EVALS_TO_RUN.get("flock"):
        logger.debug(Color.horizontal_line())
        logger.debug(Color.model_or_data_update("~~~~~ Running flock eval ~~~~~"))
        Color.in_block = True

        start_ollama_server()

        # Load GGUF into Ollama
        if not load_gguf_into_ollama(LOCAL_GGUF_FILEPATH, MODEL_NAME):
            logger.debug(Color.error("Failed to load model into Ollama. Exiting."))
            sys.exit(1)

        # Warm up the model
        warmup_ollama_model(MODEL_NAME)

        # Create database
        create_duckdb_database()

        # Set up Flock
        system_name = "flock"
        flock_conn = duckdb.connect(DUCKDB_DB_PATH)
        flock_version = "7f1c36a"
        logger.debug(Color.update(f"Installing Flock version {flock_version}..."))
        flock_conn.install_extension(
            "flock", repository="community", version=flock_version
        )
        flock_conn.load_extension("flock")

        flock_conn.execute(
            """
            CREATE SECRET (
                TYPE OPENAI,
                BASE_URL 'http://localhost:11434/v1',
                API_KEY 'N.A.'
            )
        """
        )

        flock_conn.execute(
            f"""
        CREATE MODEL(
            '{MODEL_NAME}',
            '{MODEL_NAME}', 
            'openai',
            {{"tuple_format": "JSON", "batch_size": 32, "model_parameters": {{"temperature": 0.7}}}}
        )
        """
        )

        for query_file, query_name in iter_queries("flockmtl"):
            logger.debug(Color.update(f"Running flock {query_name}..."))
            query = open(query_file).read().replace("<<model_name>>", MODEL_NAME)
            res: pd.DataFrame = flock_conn.execute(query).df()
            print(res)

        flock_conn.close()
        stop_ollama_server()
        Color.in_block = False

    if EVALS_TO_RUN.get("blendsql"):
        from blendsql import BlendSQL
        from blendsql.models import LlamaCpp

        logger.debug(Color.horizontal_line())
        logger.debug(Color.model_or_data_update("~~~~~ Running blendsql eval ~~~~~"))
        Color.in_block = True

        create_duckdb_database()

        bsql = BlendSQL(
            DUCKDB_DB_PATH,
            model=LlamaCpp(
                filename=LOCAL_GGUF_FILEPATH,
                config={
                    "n_gpu_layers": -1,
                    "n_ctx": 8000,
                    "seed": 100,
                    "n_threads": 6,
                },
                caching=False,
            ),
            verbose=True,
        )
        _ = bsql.model.model_obj

        for query_file, query_name in iter_queries("blendsql"):
            smoothie = bsql.execute(open(query_file).read())
            smoothie.print_summary()
            res: pd.DataFrame = smoothie.df
            print(query_name)
            print(res)

    if EVALS_TO_RUN.get("thalamusdb"):
        import json

        from tdb.data.relational import Database
        from tdb.execution.constraints import Constraints
        from tdb.execution.engine import ExecutionEngine
        from tdb.queries.query import Query

        logger.debug(Color.horizontal_line())
        logger.debug(Color.model_or_data_update("~~~~~ Running thalamusdb eval ~~~~~"))
        Color.in_block = True

        start_ollama_server()

        model_config = {
            "models": [
                {
                    "modalities": ["text"],
                    "priority": 10,
                    "kwargs": {
                        "filter": {
                            "model": f"ollama/{MODEL_NAME}",
                            "temperature": 0,
                            "max_tokens": 1,
                            "reasoning_effort": "disable",
                        },
                        "join": {
                            "model": f"ollama/{MODEL_NAME}",
                            "temperature": 0,
                            "stop": ["."],
                            "reasoning_effort": "disable",
                        },
                    },
                }
            ]
        }

        model_config_path = "./thalamus_db_model_config.json"

        # Create a temporary file with the model config
        with open(model_config_path, "w") as f:
            json.dump(model_config, f)

        db = Database(DUCKDB_DB_PATH)
        engine = ExecutionEngine(db=db, dop=20, model_config_path=model_config_path)
        constraints = Constraints(max_calls=1000, max_seconds=6000)
        for query_file, query_name in iter_queries("thalamusdb"):
            query = Query(db, open(query_file).read())
            result, counters = engine.run(query, constraints)
            print(query_name)
