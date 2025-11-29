from pathlib import Path
import pandas as pd
import json
import typing as t
from functools import lru_cache
import duckdb
from dataclasses import dataclass, field, asdict

import torch

from tag_queries import BLENDSQL_ANNOTATED_TAG_DATASET

from blendsql import BlendSQL
from blendsql.models import LlamaCpp, TransformersLLM
from blendsql.ingredients import LLMQA, LLMMap, LLMJoin

CURR_DIR = Path(__file__).resolve().parent


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
def create_duckdb_udfs(conn: duckdb.DuckDBPyConnection) -> None:
    def _run_llmmap(return_type: str):
        @lru_cache(maxsize=10000)
        def run_llmmap(question: str, value: str, options: list[str], quantifier: str):
            if options is not None:
                options = [i for i in set(options) if i is not None]
                options = [option.strip("'").strip('"') for option in options]
            global NUM_VALUES_PASSED
            mapped_values = LLMMap.run(
                model=model,
                question=question,
                values=[value],
                options=options,
                list_options_in_prompt=True,
                return_type=return_type,
                context_formatter=LLMMap.context_formatter,
                quantifier=quantifier,
            )
            NUM_VALUES_PASSED += len(mapped_values)
            return mapped_values[0]

        return run_llmmap

    def _run_llmqa(return_type: str):
        def run_llmqa(question: str, context: str, options: list[str], quantifier: str):
            if options is not None:
                options = [i for i in set(options) if i is not None]
                options = [option.strip("'").strip('"') for option in options]
            if context is not None:
                context = [pd.DataFrame({"values": context.split("\n---\n")})]
            response: str = LLMQA.run(
                model=model,
                question=question,
                context=context,
                options=options,
                list_options_in_prompt=True,
                return_type=return_type,
                context_formatter=LLMQA.context_formatter,
                quantifier=quantifier,
            )
            print(response)
            if return_type == "str":
                return response.strip("'").strip('"')

        return run_llmqa

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
        experiment_name="orient_dict_fmt_code_map_prompt",
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

    load_bsql = lambda path: BlendSQL(
        path,
        model=model,
        ingredients=ingredients,
        verbose=False,
    )

    prediction_data = []
    for item in BLENDSQL_ANNOTATED_TAG_DATASET:
        curr_pred_data = item.copy()
        if item["BlendSQL"] is None:
            continue
        bsql = load_bsql(load_tag_db_path(item["DB used"]))

        smoothie = bsql.execute(item["BlendSQL"])
        curr_pred_data["latency"] = smoothie.meta.process_time_seconds
        curr_pred_data["completion_tokens"] = smoothie.meta.completion_tokens
        curr_pred_data["prompt_tokens"] = smoothie.meta.prompt_tokens
        flattened_preds = [str(i) for i in smoothie.df.values.flat]
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
    prediction_df["correct"] = prediction_df["prediction"] == prediction_df["Answer"]

    output_dir = (
        CURR_DIR / f"results/TAG-Benchmark/{CONFIG.filename}/{CONFIG.experiment_name}"
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
