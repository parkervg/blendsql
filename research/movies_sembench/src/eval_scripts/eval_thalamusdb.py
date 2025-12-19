def run_thalamusdb_eval():
    import json
    import pandas as pd
    import time
    import duckdb
    from contextlib import contextmanager
    import os

    # class DummyConsole:
    #     def __getattr__(self, name):
    #         return lambda *args, **kwargs: None
    #
    # import rich.console
    #
    # rich.console.Console = DummyConsole
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
        MODEL_NAME,
        LOCAL_GGUF_FILEPATH,
        THALAMUS_CONFIG_PATH,
        SYSTEM_PARAMS,
    )
    from ..database_utils import iter_queries
    from ..ollama_utils import prepare_ollama_server, stop_ollama_server

    with duckdb.connect(DUCKDB_DB_PATH, read_only=True) as con:
        logger.debug(Color.horizontal_line())
        logger.debug(Color.model_or_data_update("~~~~~ Running thalamusdb eval ~~~~~"))
        Color.in_block = True

        ########### Prepare database + model ###########
        import rich.console

        # Disable all Rich console output
        rich.console.Console.is_terminal = False

        prepare_ollama_server(gguf_filepath=LOCAL_GGUF_FILEPATH, model_name=MODEL_NAME)

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
        return pd.DataFrame(results)

    Color.in_block = False
    stop_ollama_server()
