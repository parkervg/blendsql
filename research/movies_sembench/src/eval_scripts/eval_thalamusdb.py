from ..config import ModelConfig


def run_thalamusdb_eval(model_config: ModelConfig):
    import json
    import pandas as pd
    import time
    import duckdb
    from contextlib import contextmanager
    import os
    import sys

    @contextmanager
    def suppress_stdout():
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

    from tdb.data.relational import Database
    from tdb.execution.constraints import Constraints
    from tdb.execution.engine import ExecutionEngine
    from tdb.queries.query import Query

    from blendsql.common.logger import Color, logger

    from ..config import (
        DUCKDB_DB_PATH,
        THALAMUS_CONFIG_PATH,
        SYSTEM_PARAMS,
        MODEL_PARAMS,
    )
    from ..database_utils import iter_queries
    from ..server_utils import (
        start_llama_cpp_server,
        stop_llama_cpp_server,
        LLAMA_SERVER_HOST,
        LLAMA_SERVER_PORT,
    )

    import litellm

    litellm.drop_params = True
    litellm.completion_kwargs = {
        "max_tokens": MODEL_PARAMS["max_tokens"],
        "temperature": MODEL_PARAMS["temperature"],
    }

    with duckdb.connect(DUCKDB_DB_PATH) as con:
        logger.debug(Color.horizontal_line())
        logger.debug(Color.model_or_data_update("~~~~~ Running thalamusdb eval ~~~~~"))
        Color.in_block = True

        ########### Prepare database + model ###########
        import rich.console

        # Disable all Rich console output
        rich.console.Console.is_terminal = False

        start_llama_cpp_server(model_config)

        class CustomDatabase(Database):
            def __init__(self, con):
                self.con = con
                self.db_path = "N.A."

        #################################################

        # Create model configuration file
        with open(THALAMUS_CONFIG_PATH, "w") as f:
            json.dump(
                {
                    "models": [
                        {
                            "modalities": ["text"],
                            "priority": 10,
                            "kwargs": {
                                "filter": {
                                    "model": "openai/local-model",
                                    "api_base": f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}/v1",
                                    "api_key": "N.A.",
                                    "temperature": MODEL_PARAMS["temperature"],
                                    "max_tokens": 1,
                                    "reasoning_effort": "disable",
                                },
                                "join": {
                                    "model": "openai/local-model",
                                    "api_base": f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}/v1",
                                    "api_key": "N.A.",
                                    "temperature": MODEL_PARAMS["temperature"],
                                    "stop": ["."],
                                    "reasoning_effort": "disable",
                                },
                            },
                        }
                    ]
                },
                f,
            )

        # Initialize ThalamusDB components
        db = CustomDatabase(con)
        engine = ExecutionEngine(
            db=db,
            dop=SYSTEM_PARAMS["batch_size"],
            model_config_path=THALAMUS_CONFIG_PATH,
        )
        constraints = Constraints(max_calls=1000, max_seconds=6000)

        # Run queries
        results = []
        for query_file, query_name in iter_queries("thalamusdb"):
            query = open(query_file).read()
            start = time.time()
            query = Query(db, query)
            with suppress_stdout():
                result, counters = engine.run(query, constraints)
            latency = time.time() - start
            results.append(
                {
                    "system_name": "thalamusdb",
                    "query_name": query_name,
                    "latency": latency,
                    "prediction": result.to_json(orient="split", index=False),
                }
            )

    Color.in_block = False
    stop_llama_cpp_server()

    return pd.DataFrame(results)
