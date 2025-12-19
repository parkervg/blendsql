def run_flock_eval():
    import duckdb
    import time
    import pandas as pd
    from blendsql.common.logger import Color, logger

    from ..config import (
        DUCKDB_DB_PATH,
        MODEL_NAME,
        LOCAL_GGUF_FILEPATH,
        FLOCK_VERSION,
        FLOCK_BATCH_SIZE,
        FLOCK_TEMPERATURE,
    )
    from ..database_utils import iter_queries
    from ..ollama_utils import prepare_ollama_server, stop_ollama_server

    with duckdb.connect(DUCKDB_DB_PATH, read_only=True) as con:
        logger.debug(Color.horizontal_line())
        logger.debug(Color.model_or_data_update("~~~~~ Running flock eval ~~~~~"))
        Color.in_block = True

        ########### Prepare database + model ###########
        prepare_ollama_server(gguf_filepath=LOCAL_GGUF_FILEPATH, model_name=MODEL_NAME)
        #################################################

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
                '{MODEL_NAME}',
                '{MODEL_NAME}', 
                'openai',
                {{"tuple_format": "JSON", "batch_size": {FLOCK_BATCH_SIZE}, "model_parameters": {{"temperature": {FLOCK_TEMPERATURE}}}}}
            )
            """
        )

        # Run queries
        results = []
        for query_file, query_name in iter_queries("flockmtl"):
            query = open(query_file).read().replace("<<model_name>>", MODEL_NAME)
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
        return pd.DataFrame(results)

    stop_ollama_server()
    Color.in_block = False
