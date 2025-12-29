from ..config import ModelConfig


def run_flock_eval(model_config: ModelConfig):
    import duckdb
    import time
    import pandas as pd
    from blendsql.common.logger import Color, logger

    from ..config import (
        DUCKDB_DB_PATH,
        MODEL_PARAMS,
        SYSTEM_PARAMS,
        FLOCK_VERSION,
        DUCKDB_SEED,
    )
    from ..database_utils import iter_queries
    from ..server_utils import (
        start_llama_cpp_server,
        stop_llama_cpp_server,
        LLAMA_SERVER_HOST,
        LLAMA_SERVER_PORT,
    )

    with duckdb.connect(DUCKDB_DB_PATH) as con:
        con.execute(f"SELECT setseed({DUCKDB_SEED})")
        logger.debug(Color.horizontal_line())
        logger.debug(Color.model_or_data_update("~~~~~ Running flock eval ~~~~~"))
        Color.in_block = True

        start_llama_cpp_server(model_config)

        # Set up Flock
        logger.debug(Color.update(f"Installing Flock version {FLOCK_VERSION}..."))
        con.install_extension("flock", repository="community", version=FLOCK_VERSION)
        con.load_extension("flock")

        # Configure Flock with OpenAI-compatible endpoint
        con.execute(
            f"""
            CREATE SECRET (
                TYPE OPENAI,
                BASE_URL 'http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}/v1',
                API_KEY 'N.A.'
            )
            """
        )

        # Create model configuration
        con.execute(
            f"""
            CREATE MODEL(
                '{model_config.model_name_or_path}',
                '{model_config.model_name_or_path}', 
                'openai',
                {{"tuple_format": "json_object", "batch_size": {SYSTEM_PARAMS['batch_size']}, "model_parameters": {{"temperature": {MODEL_PARAMS['temperature']}}}}}
            )
            """
        )

        # Run queries
        results = []
        for query_file, query_name in iter_queries("flockmtl"):
            query = (
                open(query_file)
                .read()
                .replace("<<model_name>>", model_config.model_name_or_path)
            )
            start = time.time()
            result = con.execute(query).df()
            latency = time.time() - start
            results.append(
                {
                    "system_name": "flock",
                    "query_name": query_name,
                    "latency": latency,
                    "prediction": result.to_json(orient="split", index=False),
                }
            )

    stop_llama_cpp_server()
    Color.in_block = False

    return pd.DataFrame(results)
