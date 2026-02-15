import pandas as pd
from ..config import ModelConfig


def run_palimpzest_eval(model_config: ModelConfig):
    from ..config import MODEL_PARAMS, N_PARALLEL, BASE_URL

    import litellm

    original_completion = litellm.completion

    def patched_completion(*args, **kwargs):
        litellm.drop_params = True
        kwargs["api_base"] = BASE_URL
        kwargs["supports_system_message"] = False
        kwargs["temperature"] = MODEL_PARAMS["temperature"]
        kwargs["model"] = f"hosted_vllm/{model_config.model_name_or_path}"
        kwargs.pop("reasoning_effort", None)
        return original_completion(*args, **kwargs)

    # Replace the completion function with your patched version
    litellm.completion = patched_completion

    import palimpzest as pz
    import time
    import duckdb
    import importlib

    from blendsql.common.logger import Color, logger

    from ..config import DUCKDB_DB_PATH
    from ..database_utils import iter_queries
    import importlib.util
    import os

    os.environ["OPENAI_API_KEY"] = "N.A."

    def load_module(filename):
        """Load a Python file as a module and execute its run() function."""
        spec = importlib.util.spec_from_file_location("module", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    with duckdb.connect(DUCKDB_DB_PATH) as con:
        logger.debug(Color.horizontal_line())
        logger.debug(Color.model_or_data_update("~~~~~ Running palimpzest eval ~~~~~"))
        Color.in_block = True

        # Run queries
        results = []
        for query_file, query_name in iter_queries("palimpzest"):
            pz_config = pz.QueryProcessorConfig(
                max_workers=N_PARALLEL,
                join_parallelism=N_PARALLEL,
                verbose=False,
                progress=False,
                reasoning_effort=None,
                # execution_strategy="pipelined",
                # Placeholder model with reasoning
                # Need a reasoning model due to this bug: https://github.com/mitdbg/palimpzest/issues/268
                available_models=["openai/gpt-5-2025-08-07"],
            )
            func = load_module(query_file)
            start = time.time()
            result = func.run(con, pz_config).to_df()
            latency = time.time() - start
            results.append(
                {
                    "system_name": "palimpzest",
                    "query_name": query_name,
                    "latency": latency,
                    "prediction": result.to_json(orient="split", index=False),
                }
            )
    Color.in_block = False
    return pd.DataFrame(results)
