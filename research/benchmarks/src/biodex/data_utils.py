import pandas as pd
from datasets import load_dataset


def get_dataset() -> tuple[pd.DataFrame, list[str]]:
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
    return (df, unique_reactions)
