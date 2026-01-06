import pandas as pd

if __name__ == "__main__":
    from blendsql import BlendSQL
    from blendsql.models import LlamaCpp
    from blendsql.ingredients import LLMMap
    from blendsql.search import FaissVectorStore

    model = LlamaCpp(
        model_name_or_path="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
        config={"n_gpu_layers": -1, "n_ctx": 16_000, "seed": 100, "n_threads": 12},
        caching=False,
    )
    _ = model.model_obj

    BIODEX_QUESTION = "You are a medical expert. Given the patient description of a medical article, return the ranked list of medical conditions experienced by the patient. The most relevant label occurs first in the list. Be sure to rank ALL of the inputs."
    MultiLabelMap = LLMMap.from_args(
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
        options_searcher=FaissVectorStore(
            documents=unique_reactions, model_name_or_path="intfloat/e5-base-v2", k=5
        ),
    )

    bsql = BlendSQL(
        {
            "w": df,
        },
        model=model,
        verbose=True,
        ingredients=[MultiLabelMap],
    )

    smoothie = bsql.execute(
        f"""
        SELECT patient_description, 
        {{{{
            MultiLabelMap(
                '{BIODEX_QUESTION}',
                patient_description,
                return_type='list[str]',
                quantifier='{{5}}'
            )
        }}}} AS prediction, 
        reactions_list AS ground_truth 
        FROM w ORDER BY patient_description LIMIT 50
        """
    )
    smoothie.print_summary()

    res_df = smoothie.df
    res_df = res_df[~pd.isna(res_df["prediction"])]
    print(len(res_df))
    res_df["prediction"] = res_df["prediction"].apply(lambda x: list(set(x)))

    metrics_df = calculate_metrics(res_df, compute_rank_precision)
    metrics_df.mean().to_csv(f"biodex-results.csv")
    print(metrics_df.mean())
