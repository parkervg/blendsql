from pathlib import Path
import pandas as pd
from collections import defaultdict
import json
import numpy as np

from blendsql import BlendSQL
from blendsql.models import LlamaCpp
from blendsql.ingredients import LLMQA, LLMMap, LLMJoin, RAGQA


def eval(df: pd.DataFrame, output_dir: str):
    grouped = df.groupby("Query type")
    latencies = defaultdict(list)
    corrects = defaultdict(list)
    for group_name, group_df in grouped:
        for _, row in group_df.iterrows():
            qid = row["Query ID"]

            with open(f"{output_dir}/query_{qid}.json") as f:
                data = json.load(f)
                latencies[group_name].append(data["latency"])
                if data.get("error", None):
                    corrects[group_name].append(False)
                else:
                    corrects[group_name].append(data["prediction"] == data["answer"])

    kr_grouped = df.groupby("Knowledge/Reasoning Type")
    for group_name, group_df in kr_grouped:
        for _, row in group_df.iterrows():
            qid = row["Query ID"]
            if row["Query type"] != "Aggregation":
                with open(f"{output_dir}/query_{qid}.json") as f:
                    data = json.load(f)
                    latencies[group_name].append(data["latency"])
                    if data.get("error", None):
                        corrects[group_name].append(False)
                    else:
                        corrects[group_name].append(
                            data["prediction"] == data["answer"]
                        )

    for k, v in latencies.items():
        print(f"Printing stats for {k}")
        print(f"Mean latency: {np.mean(v):.2f}")
        print(f"Avg. correct: {np.mean(corrects[k]):.2f}")


def load_tag_db_path(name: str) -> str:
    return (
        Path("./research/data/bird-sql/dev_20240627/dev_databases/")
        / name
        / f"{name}.sqlite"
    )


if __name__ == "__main__":
    model = LlamaCpp(
        "Meta-Llama-3.1-8B-Instruct.Q6_K.gguf",
        "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF",
        config={"n_gpu_layers": -1},
    )
    load_bsql = lambda path: BlendSQL(
        path, model=model, ingredients={LLMQA, LLMMap, LLMJoin, RAGQA}
    )

    tag_queries = pd.read_csv("./tag-benchmark/tag_queries.csv")

    prediction_data = []
    for _, row in tag_queries.iterrows():
        curr_pred_data = {}
        bsql = load_bsql(load_tag_db_path(row["DB used"]))
        smoothie = bsql.execute(row["blendsql_query"])
        curr_pred_data["latency"] = smoothie.meta.process_time_seconds
        curr_pred_data["completion_tokens"] = smoothie.meta.completion_tokens
        curr_pred_data["prompt_tokens"] = smoothie.meta.prompt_tokens
        curr_pred_data["prediction"] = smoothie.df.to_dict()
        prediction_data.append(curr_pred_data)

    # Among the schools with the average score in Math over 560 in the SAT test, how many schools are in the bay area?
    smoothie = bsql.execute(
        """
        SELECT COUNT(DISTINCT s.CDSCode) 
            FROM schools s 
            JOIN satscores sa ON s.CDSCode = sa.cds 
            WHERE sa.AvgScrMath > 560 
            AND s.County IN {{RAGQA('Which counties are in the Bay Area?')}}
        """
    )
    print(smoothie.df)
    print(smoothie.meta.process_time_seconds)
    # blendsql_query = """
    # SELECT s.Phone
    #     FROM satscores ss
    #     JOIN schools s ON ss.cds = s.CDSCode
    #     WHERE county IN {{
    #         LLMQA(
    #             'Which counties are in the Bay Area?',
    #             (
    #                 SELECT {{
    #                     BingWebSearch('Counties in the Bay Area')
    #                 }} AS "Search Results"
    #             )
    #         )
    #     }}
    #     ORDER BY ss.AvgScrRead ASC
    #     LIMIT 1
    # """
