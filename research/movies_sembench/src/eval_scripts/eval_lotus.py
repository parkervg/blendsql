import pandas as pd

from ..config import ModelConfig, N_PARALLEL


def run_lotus_eval(model_config: ModelConfig):
    from ..config import MODEL_PARAMS
    import litellm

    original_completion = litellm.completion

    def patched_completion(*args, **kwargs):
        litellm.drop_params = True
        kwargs["supports_system_message"] = False
        kwargs["temperature"] = MODEL_PARAMS["temperature"]

        return original_completion(*args, **kwargs)

    # Replace the completion function with your patched version
    litellm.completion = patched_completion

    import time
    import duckdb
    import importlib

    import lotus
    from lotus.models import LM
    from blendsql.common.logger import Color, logger

    from ..config import DUCKDB_DB_PATH, BASE_URL, DUCKDB_SEED
    from ..gpu_util_tracker import track_gpu
    from ..database_utils import iter_queries
    import importlib.util

    def load_module(filename):
        """Load a Python file as a module and execute its run() function."""
        spec = importlib.util.spec_from_file_location("module", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    with duckdb.connect(DUCKDB_DB_PATH) as con:
        con.execute(f"SELECT setseed({DUCKDB_SEED})")
        logger.debug(Color.horizontal_line())
        logger.debug(Color.model_or_data_update("~~~~~ Running lotus eval ~~~~~"))
        Color.in_block = True

        lotus.settings.configure(
            lm=LM(
                model=f"hosted_vllm/{model_config.model_name_or_path}",
                api_base=BASE_URL,
                api_key="N.A.",
                # https://docs.litellm.ai/docs/providers/openai_compatible#advanced---disable-system-messages
                supports_system_message=False,  # lotus uses system prompts. Gemma3 doesn't listen to those.
                temperature=MODEL_PARAMS["temperature"],
                max_tokens=MODEL_PARAMS["max_tokens"],
                max_batch_size=N_PARALLEL,
            )
        )

        # Run queries
        results = []
        all_gpu_data = []
        for query_file, query_name in iter_queries("lotus"):
            lotus.settings.lm.reset_stats()
            func = load_module(query_file)
            start = time.time()
            with track_gpu() as gpu_data:
                result = func.run(con)
            all_gpu_data.append(gpu_data.copy())
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

    Color.in_block = False
    pd.DataFrame(all_gpu_data).to_csv("lotus_gpu_usage.csv", index=False)
    return pd.DataFrame(results)
