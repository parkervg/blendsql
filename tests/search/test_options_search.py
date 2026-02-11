import pytest
from datasets import load_dataset
import sys

from blendsql import BlendSQL
from blendsql.ingredients import LLMMap
from blendsql.search import FaissVectorStore, HybridSearch

BIODEX_QUESTION = "You are a medical expert. Given the patient description of a medical article, return the ranked list of medical conditions experienced by the patient. The most relevant label occurs first in the list. Be sure to rank ALL of the inputs."

FEW_SHOT_EXAMPLES = [
    {
        "question": BIODEX_QUESTION,
        "mapping": {
            "Patient experienced severe nausea and vomiting after taking the prescribed medication. The symptoms started within 2 hours of administration and persisted for 24 hours.": [
                "nausea",
                "vomiting",
                "gastrointestinal distress",
            ],
            # ... rest of your examples
        },
        "column_name": "patient_description",
        "table_name": "w",
        "return_type": "list[str]",
    }
]


@pytest.fixture(scope="session")
def biodex_data():
    """Load and prepare the dataset once per session."""
    df = load_dataset("BioDEX/BioDEX-Reactions", split="test").to_pandas()
    df["reactions_list"] = df["reactions"].apply(lambda x: x.split(","))
    df["reactions_list"] = df["reactions_list"].apply(
        lambda x: [r.strip().lower() for r in x]
    )
    df["num_labels"] = df["reactions_list"].apply(lambda x: len(x))
    df["patient_description"] = df["fulltext_processed"].apply(lambda x: x[:8000])
    unique_reactions = list(
        set([item for sublist in df["reactions_list"].tolist() for item in sublist])
    )
    return df, unique_reactions


@pytest.fixture(params=["faiss", "hybrid"])
def bsql(request, biodex_data):
    df, unique_reactions = biodex_data
    searcher_type = request.param

    if searcher_type == "faiss":
        searcher = FaissVectorStore(
            documents=unique_reactions,
            model_name_or_path="sentence-transformers/all-mpnet-base-v2",
            k=5,
        )
    elif searcher_type == "hybrid":
        searcher = HybridSearch(
            documents=unique_reactions,
            model_name_or_path="sentence-transformers/all-mpnet-base-v2",
            k=5,
        )

    MultiLabelMap = LLMMap.from_args(
        few_shot_examples=FEW_SHOT_EXAMPLES,
        options_searcher=searcher,
    )

    return BlendSQL(
        {"w": df},
        ingredients=[MultiLabelMap],
    )


@pytest.mark.skipif(sys.platform == "darwin", reason="faiss tests skipped on MacOS")
def test_options_search(bsql, model):
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
        model=model,
    )
