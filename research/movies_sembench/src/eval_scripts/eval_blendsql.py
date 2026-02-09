from ..config import ModelConfig


def run_blendsql_eval(model_config: ModelConfig):
    import pandas as pd
    import time
    import duckdb

    from blendsql import BlendSQL
    from blendsql.models import VLLM
    from blendsql.db import DuckDB
    from blendsql.common.logger import Color, logger
    from blendsql import config

    from ..config import DUCKDB_DB_PATH, N_PARALLEL, BASE_URL, DUCKDB_SEED, MODEL_PARAMS
    from ..database_utils import iter_queries

    config.set_deterministic(True)
    config.set_async_limit(N_PARALLEL)
    config.set_default_max_tokens(MODEL_PARAMS["max_tokens"])

    try:
        with duckdb.connect(DUCKDB_DB_PATH) as con:
            con.execute(f"SELECT setseed({DUCKDB_SEED})")
            logger.debug(Color.horizontal_line())
            logger.debug(
                Color.model_or_data_update("~~~~~ Running blendsql eval ~~~~~")
            )
            Color.in_block = True

            # start_vllm_server(model_config)

            # Initialize BlendSQL
            bsql = BlendSQL(
                DuckDB(con),
                model=VLLM(
                    model_name_or_path=model_config.model_name_or_path,
                    base_url=BASE_URL,
                ),
                verbose=False,
            )

            # Warmup
            _ = bsql.execute(
                """
                SELECT {{LLMQA('What color is the sky?')}} AS answer
                """
            )
            _ = bsql.execute(
                """
                WITH subset AS (
                    SELECT * FROM Reviews LIMIT 1
                )
                SELECT {{LLMMap('Say hello', reviewText)}} FROM subset
                """
            )

            # Run queries
            results = []
            for query_file, query_name in iter_queries("blendsql"):
                query = open(query_file).read()
                start = time.time()
                smoothie = bsql.execute(query)
                result = (
                    smoothie.df
                )  # Count this, since conversion to pd from pl takes a small bit of latency
                latency = time.time() - start
                results.append(
                    {
                        "system_name": "blendsql",
                        "query_name": query_name,
                        "latency": latency,
                        "prediction": result.to_json(orient="split", index=False),
                        "num_generation_calls": smoothie.meta.num_generation_calls,
                        "output_tokens": smoothie.meta.completion_tokens,
                        "input_tokens": smoothie.meta.prompt_tokens,
                    }
                )
        Color.in_block = False
        return pd.DataFrame(results)
    finally:
        # stop_vllm_server()
        pass
