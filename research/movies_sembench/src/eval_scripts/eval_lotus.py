import pandas as pd

from ..config import ModelConfig


def run_lotus_eval(model_config: ModelConfig):
    import time
    import duckdb
    import importlib

    import lotus
    from lotus.models import LM
    from blendsql.common.logger import Color, logger

    from ..config import DUCKDB_DB_PATH, SYSTEM_PARAMS, MODEL_PARAMS, DUCKDB_SEED
    from ..server_utils import (
        start_llama_cpp_server,
        stop_llama_cpp_server,
        LLAMA_SERVER_HOST,
        LLAMA_SERVER_PORT,
    )
    from ..database_utils import iter_queries
    import importlib.util

    import litellm

    litellm.drop_params = True
    litellm.completion_kwargs = {
        "max_tokens": MODEL_PARAMS["max_tokens"],
        "temperature": MODEL_PARAMS["temperature"],
    }

    def load_module(filename):
        """Load a Python file as a module and execute its run() function."""
        spec = importlib.util.spec_from_file_location("module", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    start_llama_cpp_server(model_config)

    with duckdb.connect(DUCKDB_DB_PATH) as con:
        con.execute(f"SELECT setseed({DUCKDB_SEED})")
        logger.debug(Color.horizontal_line())
        logger.debug(Color.model_or_data_update("~~~~~ Running lotus eval ~~~~~"))
        Color.in_block = True

        lotus.settings.configure(
            lm=LM(
                model=f"openai/local-model",
                api_base=f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}/v1",
                api_key="N.A.",
                # https://docs.litellm.ai/docs/providers/openai_compatible#advanced---disable-system-messages
                supports_system_message=False,  # lotus uses system prompts. Gemma3 doesn't listen to those.
                temperature=MODEL_PARAMS["temperature"],
                max_batch_size=SYSTEM_PARAMS["batch_size"],
                max_tokens=MODEL_PARAMS["max_tokens"],
            )
        )

        # Run queries
        results = []
        for query_file, query_name in iter_queries("lotus"):
            lotus.settings.lm.reset_stats()
            func = load_module(query_file)
            start = time.time()
            result = func.run(con)
            latency = time.time() - start
            results.append(
                {
                    "system_name": "lotus",
                    "query_name": query_name,
                    "latency": latency,
                    "prediction": result.to_json(orient="split", index=False),
                    "num_generation_calls": "N.A.",
                    "output_tokens": lotus.settings.lm.stats.physical_usage.completion_tokens,
                    "input_tokens": lotus.settings.lm.stats.physical_usage.prompt_tokens,
                }
            )

    stop_llama_cpp_server()
    Color.in_block = False
    return pd.DataFrame(results)
