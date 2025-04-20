from pathlib import Path
import pandas as pd
import json
import typing as t
from dataclasses import dataclass, field, asdict

from tag_queries import TAG_DATASET

from blendsql import BlendSQL
from blendsql.models import LlamaCpp, TransformersLLM
from blendsql.ingredients import LLMQA, LLMMap, LLMJoin, RAGQA

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
        experiment_name="no_llmqa_examples",
    )

    # CONFIG = ExperimentConfig(
    #     repo_id="meta-llama/Llama-3.2-1B-Instruct",
    #     experiment_name="no_llmqa_examples"
    # )
    ingredients = {
        LLMQA.from_args(k=0),
        LLMMap,
        LLMJoin,
        RAGQA,
    }

    if CONFIG.filename is not None:
        model = LlamaCpp(
            CONFIG.filename,
            CONFIG.repo_id,
            config={"n_gpu_layers": -1, "n_ctx": 9600},
            caching=False,
        )
    else:
        model = TransformersLLM(
            CONFIG.repo_id, config={"device_map": "auto"}, caching=True
        )
    # Pre-load model obj
    _ = model.model_obj

    load_bsql = lambda path: BlendSQL(
        path,
        model=model,
        ingredients=ingredients,
        verbose=True,
    )

    prediction_data = []
    for item in TAG_DATASET:
        curr_pred_data = item.copy()
        if item["BlendSQL"] is None:
            continue
        if item["Query ID"] != 18:
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
