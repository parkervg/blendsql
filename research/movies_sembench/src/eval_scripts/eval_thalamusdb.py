import tdb.operators.semantic_filter
import litellm
from litellm import completion
from ..config import MODEL_PARAMS


def make_llama_compatible(config):
    """
    Convert OpenAI multi-part content format to llama-cpp-server compatible format.

    Args:
        config: Dictionary containing the API configuration and messages

    Returns:
        Modified dict with compatible message format
    """
    # Create a deep copy to avoid modifying the original
    new_config = config.copy()

    if "messages" in new_config:
        new_messages = []
        for message in new_config["messages"]:
            new_message = message.copy()
            # Check if content is a list (multi-part format)
            if isinstance(new_message.get("content"), list):
                # Extract and concatenate all text parts
                text_parts = []
                for part in new_message["content"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                # Join with newline or space (you can adjust the separator)
                new_message["content"] = "\n".join(text_parts)
            new_messages.append(new_message)
        new_config["messages"] = new_messages
    return new_config


def _modified_filter_completion_wrapper(item_text, kwargs):
    """Invoke completion function with given keyword arguments.

    Args:
        item_text (str): Text representation of the item.
        kwargs (dict): Keyword arguments for the completion function.

    Returns:
        tuple: (item_text, kwargs, LLM response).
    """
    # Ensure parameters are dropped for logging where applicable
    litellm.drop_params = True
    kwargs["temperature"] = MODEL_PARAMS["temperature"]
    kwargs["supports_system_message"] = False
    response = completion(**make_llama_compatible(kwargs))
    # ThalamusDB does this on their filter:
    # `results.append((item_text, result == '1'))`
    # Many times small models will add a leading/trailing newline, making this equality wrong.
    # We patch that here.
    model_response = response.choices[0].message.content
    response.choices[0].message.content = model_response.strip()
    return item_text, kwargs, response


tdb.operators.semantic_filter._filter_completion_wrapper = (
    _modified_filter_completion_wrapper
)

from tdb.operators.semantic_join import BatchJoin


class CustomBatchJoin(BatchJoin):
    def _find_matches(self, pairs):
        """Finds pairs satisfying the join condition.

        Args:
            pairs: List of key pairs to check for matches.

        Returns:
            list: List of key pairs that satisfy the join condition.
        """
        # Get list of unique keys from both tables
        left_keys = sorted(set(left_key for left_key, _ in pairs))
        right_keys = sorted(set(right_key for _, right_key in pairs))
        # Prepare the items for the LLM prompt
        left_items = [self._encode_item(left_key) for left_key in left_keys]
        right_items = [self._encode_item(right_key) for right_key in right_keys]
        # If there are no keys, return empty list
        nr_left_items = len(left_items)
        nr_right_items = len(right_items)
        if nr_left_items == 0 or nr_right_items == 0:
            return []
        # print(f'Nr of left items: {nr_left_items}, ')
        # print(f'Nr of right items: {nr_right_items}')
        # Construct prompt for LLM
        prompt = self._create_prompt(left_items, right_items)
        messages = [prompt]
        base = self._best_model_args(messages)["join"]
        kwargs = {**base, "messages": messages}
        litellm.drop_params = True
        kwargs["temperature"] = MODEL_PARAMS["temperature"]
        kwargs["supports_system_message"] = False
        response = completion(**make_llama_compatible(kwargs))
        model = kwargs["model"]
        self.update_cost_counters(model, response)
        matching_keys = []
        try:
            matching_keys = self._extract_matches(left_keys, right_keys, response)
        except:
            print("Incorrect output format in LLM reply - continuing join.")
            # traceback.print_exc()

        return matching_keys


tdb.operators.semantic_join.BatchJoin = CustomBatchJoin

from tdb.execution.counters import LLMCounters

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
        DUCKDB_SEED,
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
        con.execute(f"SELECT setseed({DUCKDB_SEED})")
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
        tdb_model_name = "openai/local-model"
        with open(THALAMUS_CONFIG_PATH, "w") as f:
            json.dump(
                {
                    "models": [
                        {
                            "modalities": ["text"],
                            "priority": 10,
                            "kwargs": {
                                "filter": {
                                    "model": tdb_model_name,
                                    "api_base": f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}/v1",
                                    "api_key": "N.A.",
                                    "temperature": MODEL_PARAMS["temperature"],
                                    "max_tokens": 1,
                                    "reasoning_effort": "disable",
                                },
                                "join": {
                                    "model": tdb_model_name,
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
            model_counter: LLMCounters = counters.model2counters[tdb_model_name]
            results.append(
                {
                    "system_name": "thalamusdb",
                    "query_name": query_name,
                    "latency": latency,
                    "prediction": result.to_json(orient="split", index=False),
                    "num_generation_calls": model_counter.LLM_calls,
                    "output_tokens": model_counter.output_tokens,
                    "input_tokens": model_counter.input_tokens,
                }
            )

    Color.in_block = False
    stop_llama_cpp_server()

    return pd.DataFrame(results)
