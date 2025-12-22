def run_flock_eval():
    import duckdb
    import time
    import pandas as pd
    from blendsql.common.logger import Color, logger

    from ..config import (
        DUCKDB_DB_PATH,
        MODEL_PARAMS,
        SYSTEM_PARAMS,
        FLOCK_VERSION,
        MODEL_CONFIG,
    )
    from ..database_utils import iter_queries
    from ..ollama_utils import prepare_ollama_server, stop_ollama_server

    with duckdb.connect(DUCKDB_DB_PATH, read_only=True) as con:
        logger.debug(Color.horizontal_line())
        logger.debug(Color.model_or_data_update("~~~~~ Running flock eval ~~~~~"))
        Color.in_block = True

        prepare_ollama_server(model_name=MODEL_CONFIG.ollama_model_name)

        # Set up Flock
        logger.debug(Color.update(f"Installing Flock version {FLOCK_VERSION}..."))
        con.install_extension("flock", repository="community", version=FLOCK_VERSION)
        con.load_extension("flock")

        # Configure Flock with OpenAI-compatible endpoint
        con.execute(
            """
            CREATE SECRET (
                TYPE OPENAI,
                BASE_URL 'http://localhost:11434/v1',
                API_KEY 'N.A.'
            )
            """
        )

        # Create model configuration
        con.execute(
            f"""
            CREATE MODEL(
                '{MODEL_CONFIG.ollama_model_name}',
                '{MODEL_CONFIG.ollama_model_name}', 
                'openai',
                {{"tuple_format": "JSON", "batch_size": {SYSTEM_PARAMS['batch_size']}, "model_parameters": {{"temperature": {MODEL_PARAMS['temperature']}}}}}
            )
            """
        )

        # Run queries
        results = []
        for query_file, query_name in iter_queries("flockmtl"):
            query = (
                open(query_file)
                .read()
                .replace("<<model_name>>", MODEL_CONFIG.ollama_model_name)
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

    stop_ollama_server()
    Color.in_block = False

    return pd.DataFrame(results)
