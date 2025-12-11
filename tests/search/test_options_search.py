import pytest
from datasets import load_dataset
import sys

from blendsql import BlendSQL
from blendsql.ingredients import LLMMap
from blendsql.search import FaissVectorStore

BIODEX_QUESTION = "You are a medical expert. Given the patient description of a medical article, return the ranked list of medical conditions experienced by the patient. The most relevant label occurs first in the list. Be sure to rank ALL of the inputs."


@pytest.fixture(scope="session")
def bsql() -> BlendSQL:
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
            documents=unique_reactions,
            model_name_or_path="sentence-transformers/all-mpnet-base-v2",
            k=5,
        ),
    )
    return BlendSQL(
        {
            "w": df,
        },
        ingredients=[MultiLabelMap],
    )


@pytest.mark.skipif(sys.platform == "darwin", reason="faiss tests skipped on MacOS")
def test_faiss_options_search(bsql, constrained_model):
    _ = bsql.execute(
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
            FROM w ORDER BY patient_description LIMIT 5
            """,
        model=constrained_model,
    )
