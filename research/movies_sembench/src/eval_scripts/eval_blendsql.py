from ..config import ModelConfig


def run_blendsql_eval(model_config: ModelConfig):
    import pandas as pd
    import time
    import json
    from textwrap import indent
    import duckdb

    from blendsql import BlendSQL
    from blendsql.models import LlamaCpp
    from blendsql.db import DuckDB
    from blendsql.ingredients import LLMMap
    from blendsql.common.logger import Color, logger
    from blendsql.configure import set_default_max_tokens

    from ..config import (
        DUCKDB_DB_PATH,
        SYSTEM_PARAMS,
        MODEL_PARAMS,
    )
    from ..database_utils import iter_queries

    set_default_max_tokens(MODEL_PARAMS["max_tokens"])

    with duckdb.connect(DUCKDB_DB_PATH) as con:
        logger.debug(Color.horizontal_line())
        logger.debug(Color.model_or_data_update("~~~~~ Running blendsql eval ~~~~~"))
        Color.in_block = True

        # Initialize BlendSQL
        config = {
            "n_gpu_layers": -1,
            "n_ctx": MODEL_PARAMS["num_ctx"],
            "seed": MODEL_PARAMS["seed"],
            "n_threads": MODEL_PARAMS["num_threads"],
            "temperature": MODEL_PARAMS["temperature"],
            "flash_attn": MODEL_PARAMS["flash_attn"],
        }
        logger.debug(
            Color.model_or_data_update(
                f"Using config: {indent(json.dumps(config, indent=4), '    ')}"
            )
        )
        bsql = BlendSQL(
            DuckDB(con),
            model=LlamaCpp(
                model_name_or_path=model_config.model_name_or_path,
                filename=model_config.filename,
                config={
                    "n_gpu_layers": -1,
                    "n_ctx": MODEL_PARAMS["num_ctx"],
                    "seed": MODEL_PARAMS["seed"],
                    "n_threads": MODEL_PARAMS["num_threads"],
                    "temperature": MODEL_PARAMS["temperature"],
                    "flash_attn": MODEL_PARAMS["flash_attn"],
                },
                caching=False,
            ),
            ingredients={LLMMap.from_args(batch_size=SYSTEM_PARAMS["batch_size"])},
            verbose=False,
        )

        # Initialize model
        _ = bsql.model.model_obj

        # Run queries
        results = []
        for query_file, query_name in iter_queries("blendsql"):
            query = open(query_file).read()
            start = time.time()
            result = bsql.execute(query).df
            latency = time.time() - start
            results.append(
                {
                    "system_name": "blendsql",
                    "query_name": query_name,
                    "latency": latency,
                    "prediction": result.to_json(orient="split", index=False),
                }
            )

    Color.in_block = False
    return pd.DataFrame(results)
