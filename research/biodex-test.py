from datasets import load_dataset
import pandas as pd
from typing import Callable

# import lotus
# from lotus.models import LM, LiteLLMRM
# from lotus.types import CascadeArgs
# from lotus.vector_store import FaissVS


def compute_recall(gt_ids, ids, cutoff=1000):
    return len(set(gt_ids).intersection(set(ids[:cutoff]))) / len(gt_ids)


def compute_precision(gt_ids, ids, cutoff=1000):
    if len(ids[:cutoff]) == 0:
        return 0
    else:
        return len(set(gt_ids).intersection(set(ids[:cutoff]))) / len(ids[:cutoff])


def compute_rank_precision(gt_ids, ids, cutoff=1000):
    if len(ids[:cutoff]) == 0:
        return 0
    else:
        divisor = min(len(gt_ids), cutoff)
        count = 0
        for i in range(min(cutoff, len(ids))):
            if ids[i] in gt_ids:
                count += 1
        return count / divisor


if __name__ == "__main__":
    df = load_dataset("BioDEX/BioDEX-Reactions", split="test").to_pandas()

    # split and remove trailing or leading whitespace
    df["reactions_list"] = df["reactions"].apply(lambda x: x.split(","))
    df["reactions_list"] = df["reactions_list"].apply(
        lambda x: [r.strip().lower() for r in x]
    )
    df["num_labels"] = df["reactions_list"].apply(lambda x: len(x))

    # truncate the fulltext to 8000 chars
    df["patient_description"] = df["fulltext_processed"].apply(lambda x: x[:8000])
    unique_reactions = list(
        set([item for sublist in df["reactions_list"].tolist() for item in sublist])
    )

    from blendsql import BlendSQL
    from blendsql.models import LlamaCpp
    from blendsql.ingredients import LLMMap
    from blendsql.search import HybridSearch

    model = LlamaCpp(
        model_name_or_path="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        config={"n_gpu_layers": -1, "n_ctx": 16_000, "seed": 100, "n_threads": 12},
    )
    _ = model.model_obj
    BIODEX_QUESTION = "Given the patient description of a medical article, what are the adverse drug reactions that are likely affecting the patient, as a list sorted from most confident -> least confident?"

    for options_k in [
        # 10,
        20,
        # 30,
        # 40,
        # 50
    ]:
        bsql = BlendSQL(
            {
                "w": df,
                "reactions": pd.DataFrame({"name": unique_reactions}),
            },
            model=model,
            verbose=True,
            ingredients=[
                LLMMap.from_args(
                    few_shot_examples=[
                        {
                            "question": BIODEX_QUESTION,
                            "mapping": {
                                "Patient experienced severe nausea and vomiting after taking the prescribed medication. The symptoms started within 2 hours of administration and persisted for 24 hours.": [
                                    "nausea",
                                    "vomiting",
                                    "gastrointestinal distress",
                                ],
                                "Subject reported persistent headache and dizziness following drug treatment. These symptoms interfered with daily activities and lasted for several days.": [
                                    "headache",
                                    "dizziness",
                                    "neurological symptoms",
                                ],
                                "Individual developed widespread skin rash and intense itching after medication use. The reaction appeared on arms, torso, and face within hours of taking the drug.": [
                                    "skin rash",
                                    "itching",
                                    "allergic reaction",
                                    "dermatological symptoms",
                                ],
                                "Patient complained of severe stomach pain and diarrhea after taking the medication. The gastrointestinal symptoms were debilitating and required medical attention.": [
                                    "stomach pain",
                                    "diarrhea",
                                    "gastrointestinal symptoms",
                                    "abdominal distress",
                                ],
                                "Subject experienced extreme fatigue and muscle weakness following medication administration. Energy levels remained critically low for 48-72 hours post-treatment.": [
                                    "fatigue",
                                    "weakness",
                                    "muscle weakness",
                                    "energy depletion",
                                ],
                            },
                            "column_name": "patient_description",
                            "table_name": "w",
                            "return_type": "list[str]",
                        }
                    ],
                    option_searcher=lambda d: HybridSearch(
                        documents=d,
                        model_name_or_path="intfloat/e5-base-v2",
                        k=20,
                    ),
                )
            ],
        )

        smoothie = bsql.execute(
            f"""
            SELECT {{{{
                LLMMap(
                    '{BIODEX_QUESTION}',
                    patient_description,
                    options=(SELECT DISTINCT name FROM reactions),
                    return_type='list[str]',
                    quantifier='{{1,10}}'
                )
            }}}} AS "prediction", reactions_list AS "ground_truth" FROM w LIMIT 250
            """
        )

        res_df = smoothie.df
        res_df["prediction"] = res_df["prediction"].apply(lambda x: list(set(x)))

        def calculate_metrics(df: pd.DataFrame, f: Callable):
            df["rank_precision@5"] = df.apply(
                lambda x: f(x["ground_truth"], x["prediction"], cutoff=5),
                axis=1,
            )
            df["rank-precision@10"] = df.apply(
                lambda x: f(x["ground_truth"], x["prediction"], cutoff=10),
                axis=1,
            )

            df["rank-precision@25"] = df.apply(
                lambda x: f(x["ground_truth"], x["prediction"], cutoff=25),
                axis=1,
            )

            df["num_ids"] = df.apply(lambda x: len(x["prediction"]), axis=1)

            # take subset of df with metrics
            return df[
                [
                    col
                    for col in df.columns
                    if "@" in col or "latency" in col or "num_ids" in col
                ]
            ]

        metrics_df = calculate_metrics(res_df, compute_rank_precision)
        metrics_df.mean().to_csv(f"biodex-results-{options_k}.csv")
        print(options_k)
        print(metrics_df.mean())
