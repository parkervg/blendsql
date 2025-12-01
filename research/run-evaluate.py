import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import json
import typing as t
import duckdb
import pyarrow as pa
from dataclasses import dataclass, field, asdict

import torch

from tag_queries import BLENDSQL_ANNOTATED_TAG_DATASET

from blendsql import BlendSQL
from blendsql.models import LlamaCpp, TransformersLLM
from blendsql.ingredients import LLMQA, LLMMap, LLMJoin
from blendsql.ingredients import Ingredient

CURR_DIR = Path(__file__).resolve().parent
NUM_VALUES_PASSED = 0


@dataclass
class ExperimentConfig:
    repo_id: str = field()
    experiment_name: str = field()
    filename: t.Optional[str] = field(default=None)


def load_tag_db_path(name: str) -> str:
    return (
        CURR_DIR / "data/bird-sql/dev_20240627/dev_databases/" / name / f"{name}.sqlite"
    )


# Define DuckDB UDFs
# Since UDFs here don't support a union type,
# we need to create a separate function for each possible type
# https://duckdb.org/docs/stable/clients/python/function#creating-functions
def create_duckdb_udfs(
    conn: duckdb.DuckDBPyConnection, ingredients: Iterable[t.Type[Ingredient]]
) -> None:
    name_to_initialized_ingredient = {}
    for ingredient in ingredients:
        name_to_initialized_ingredient[ingredient.__name__] = ingredient(
            name=ingredient.__name__,
            db=None,
            session_uuid=None,
        )

    def _get_pa_type(return_type: str):
        """Helper to get PyArrow type from return_type string."""
        if return_type == "bool":
            return pa.bool_()
        elif return_type == "int":
            return pa.int64()
        elif return_type in ("str", "substring"):
            return pa.string()
        elif return_type == "List[str]":
            return pa.list_(pa.string())

    def _run_llmmap(return_type: str):
        def run_llmmap(
            questions: pa.Array,
            values: pa.Array,
            options: pa.Array,
            quantifiers: pa.Array,
        ):
            global NUM_VALUES_PASSED
            # Convert Arrow arrays to Python lists
            questions_list = questions.to_pylist()
            values_list = values.to_pylist()
            quantifiers_list = quantifiers.to_pylist()

            # Handle the question and quantifier (may be None/null)
            question = questions_list[0]
            quantifier = quantifiers_list[0]  # Will be Python None if SQL NULL

            # Handle options - need to check if the scalar itself is null
            # options is an Array of Lists, so options[0] is a ListScalar
            if options[0].is_valid:  # Check if the scalar is not null
                opts = options[0].as_py()  # Convert to Python list
                opts = list(set(o.strip("'\"") for o in opts if o is not None))
            else:
                opts = None

            # Filter out null values from the input
            unique_values = list(set(v for v in values_list if v is not None))
            if not unique_values:
                # All values were null, return nulls
                return pa.array(
                    [None] * len(values_list), type=_get_pa_type(return_type)
                )
            NUM_VALUES_PASSED += len(unique_values)
            # Single batched LLM call for all unique values
            mapped_results = name_to_initialized_ingredient["LLMMap"].run(
                model=model,
                question=question,
                values=unique_values,
                options=opts,
                list_options_in_prompt=True,
                return_type=return_type,
                context_formatter=None,
                quantifier=quantifier,
            )

            # Create lookup dict
            result_map = dict(zip(unique_values, mapped_results))

            # Map back to original order, preserving nulls
            results = [
                result_map.get(v) if v is not None else None for v in values_list
            ]

            return pa.array(results, type=_get_pa_type(return_type))

        return run_llmmap

    def _run_llmqa(return_type: str):
        def run_llmqa(question: str, context: str, options: list[str], quantifier: str):
            cleaned_options = (
                [opt.strip("'\"") for opt in set(options) if opt is not None]
                if options is not None
                else None
            )

            parsed_context = (
                [pd.DataFrame({"values": context.split("\n---\n")})]
                if context is not None
                else None
            )

            response: str = name_to_initialized_ingredient["LLMQA"].run(
                model=model,
                question=question,
                context=parsed_context,
                options=cleaned_options,
                list_options_in_prompt=True,
                return_type=return_type,
                context_formatter=lambda df: json.dumps(
                    df.to_dict(orient="records"), ensure_ascii=False, indent=4
                ),
                quantifier=quantifier,
            )
            if return_type == "str":
                return response.strip("'").strip('"')
            return response

        return run_llmqa

    # https://duckdb.org/docs/stable/clients/python/function#creating-functions
    for func_name, func in [("LLMMap", _run_llmmap), ("LLMQA", _run_llmqa)]:
        conn.create_function(
            name=f"{func_name}Bool",
            function=func("bool"),
            parameters=[
                duckdb.sqltype("string"),
                duckdb.sqltype("string"),
                duckdb.list_type(duckdb.sqltype("string")),
                duckdb.sqltype("string"),
            ],
            return_type=duckdb.sqltype("bool"),
            null_handling="special",
            type="arrow" if func_name == "LLMMap" else "native",
            side_effects=False,
        )
        conn.create_function(
            name=f"{func_name}Int",
            function=func("int"),
            parameters=[
                duckdb.sqltype("string"),
                duckdb.sqltype("string"),
                duckdb.list_type(duckdb.sqltype("string")),
                duckdb.sqltype("string"),
            ],
            return_type=duckdb.sqltype("int"),
            null_handling="special",
            type="arrow" if func_name == "LLMMap" else "native",
            side_effects=False,
        )
        conn.create_function(
            name=f"{func_name}Str",
            function=func("str"),
            parameters=[
                duckdb.sqltype("string"),
                duckdb.sqltype("string"),
                duckdb.list_type(duckdb.sqltype("string")),
                duckdb.sqltype("string"),
            ],
            return_type=duckdb.sqltype("string"),
            null_handling="special",
            type="arrow" if func_name == "LLMMap" else "native",
            side_effects=False,
        )
        conn.create_function(
            name=f"{func_name}Substr",
            function=func("substring"),
            parameters=[
                duckdb.sqltype("string"),
                duckdb.sqltype("string"),
                duckdb.list_type(duckdb.sqltype("string")),
                duckdb.sqltype("string"),
            ],
            return_type=duckdb.sqltype("string"),
            null_handling="special",
            type="arrow" if func_name == "LLMMap" else "native",
            side_effects=False,
        )
        conn.create_function(
            name=f"{func_name}List",
            function=func("List[str]"),
            parameters=[
                duckdb.sqltype("string"),
                duckdb.sqltype("string"),
                duckdb.list_type(duckdb.sqltype("string")),
                duckdb.sqltype("string"),
            ],
            return_type=duckdb.list_type(duckdb.sqltype("string")),
            null_handling="special",
            type="arrow" if func_name == "LLMMap" else "native",
            side_effects=False,
        )


def get_group_stats(df: pd.DataFrame, group_on: str, metrics: list) -> pd.DataFrame:
    def _get_single_group(metric):
        metric_by_group = df.groupby(group_on)[metric].mean()
        overall_metric = df[metric].mean()
        metric_by_group["Total"] = overall_metric
        aggregated_results = pd.DataFrame(metric_by_group).T
        aggregated_results.columns.name = ""
        return aggregated_results

    return pd.concat([_get_single_group(metric) for metric in metrics])


if __name__ == "__main__":
    CONFIG = ExperimentConfig(
        repo_id="QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct.Q6_K.gguf",
        experiment_name="current",
    )

    ingredients = {
        LLMQA.from_args(
            num_few_shot_examples=0,
            context_formatter=lambda df: json.dumps(
                df.to_dict(orient="records"), ensure_ascii=False, indent=4
            ),
        ),
        LLMMap,
        LLMJoin,
    }

    # from blendsql.db import SQLite
    # all_dbs = set([item["DB used"] for item in BLENDSQL_ANNOTATED_TAG_DATASET if item["BlendSQL"] is not None])
    # num_rows = []
    # for db_path in all_dbs:
    #     db = SQLite(load_tag_db_path(db_path))
    #     for t in db.tables():
    #         num_rows.append(db.execute_to_list(f"SELECT COUNT(*) FROM {t}")[0])
    #

    if CONFIG.filename is not None:
        model = LlamaCpp(
            CONFIG.filename,
            CONFIG.repo_id,
            config={"n_gpu_layers": -1, "n_ctx": 8000, "seed": 100, "n_threads": 16},
            caching=False,
        )
    else:
        model = TransformersLLM(
            CONFIG.repo_id,
            config={"device_map": "auto", "torch_dtype": torch.bfloat16},
            caching=False,
        )
    # Pre-load model obj
    _ = model.model_obj

    for exp_type in [
        # "DuckDB",
        "BlendSQL"
    ]:
        if exp_type == "BlendSQL":
            load_bsql = lambda path: BlendSQL(
                path,
                model=model,
                ingredients=ingredients,
                verbose=True,  # toggle this off for actual runtime test
            )
        do_eval = False
        prediction_data = []
        for item in BLENDSQL_ANNOTATED_TAG_DATASET:
            if item["Answer"] is None:
                continue
            curr_pred_data = item.copy()
            if exp_type == "BlendSQL":
                if item["BlendSQL"] is None:
                    continue
                bsql = load_bsql(load_tag_db_path(item["DB used"]))
                smoothie = bsql.execute(item["BlendSQL"])
                curr_pred_data["latency"] = smoothie.meta.process_time_seconds
                curr_pred_data["completion_tokens"] = smoothie.meta.completion_tokens
                curr_pred_data["prompt_tokens"] = smoothie.meta.prompt_tokens
                curr_pred_data["num_values_passed"] = smoothie.meta.num_values_passed
                flattened_preds = [str(i) for i in smoothie.df.values.flat]
            elif exp_type == "DuckDB":
                # if not do_eval:
                #     if item["Query ID"] == 29:
                #         do_eval = True
                #     continue
                NUM_VALUES_PASSED = 0
                print(f"Running Query ID {item['Query ID']}...")
                conn = duckdb.connect()
                create_duckdb_udfs(conn, ingredients)
                conn.execute(
                    f"""ATTACH '{load_tag_db_path(item["DB used"])}' AS db (TYPE SQLITE)"""
                )
                conn.execute("SET search_path = 'db,main'")
                conn.execute("SET arrow_large_buffer_size = true")
                start = time.time()
                try:
                    pred = conn.execute(item["DuckDB"]).df()
                except Exception as e:
                    print(e)
                flattened_preds = [str(i) for i in pred.values.flat]
                curr_pred_data["completion_tokens"] = -1
                curr_pred_data["prompt_tokens"] = -1
                curr_pred_data["latency"] = time.time() - start
                curr_pred_data["num_values_passed"] = NUM_VALUES_PASSED
                print(NUM_VALUES_PASSED)
                print(flattened_preds)
                model.model_obj.engine.model_obj.reset()
            pred_to_add = flattened_preds
            if len(curr_pred_data["Answer"]) == 1:
                curr_pred_data["Answer"] = curr_pred_data["Answer"][0]
            if len(flattened_preds) > 1:
                if item.get("order_insensitive_answer", False):
                    pred_to_add = sorted(flattened_preds)
                    curr_pred_data["Answer"] = sorted(curr_pred_data["Answer"])
            else:
                pred_to_add = next(iter(flattened_preds), None)
            curr_pred_data["prediction"] = pred_to_add
            prediction_data.append(curr_pred_data)

        prediction_df = pd.DataFrame(prediction_data)
        prediction_df["correct"] = (
            prediction_df["prediction"] == prediction_df["Answer"]
        )

        output_dir = (
            CURR_DIR
            / f"results/TAG-Benchmark/{exp_type.lower()}/{CONFIG.filename}/{CONFIG.experiment_name}"
        )
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        print(f"Results for {CONFIG.filename}: {CONFIG.experiment_name}")
        metric_df = get_group_stats(
            df=prediction_df, group_on="Query type", metrics=["correct", "latency"]
        )
        print(metric_df)
        with open(output_dir / f"aggregated_metrics.csv", "w") as f:
            metric_df.to_json(f, indent=4)
        with open(output_dir / f"config.json", "w") as f:
            json.dumps(asdict(CONFIG), indent=4)
        prediction_df.to_csv(output_dir / "predictions.csv", index=False)
        torch.cuda.empty_cache()
