import typing as t
import numpy as np
import re
import requests
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tag_queries import BLENDSQL_ANNOTATED_TAG_DATASET


def get_lotus_token_counts(
    use_ids: t.List[str], tokenizer: PreTrainedTokenizerBase
) -> t.List[str]:
    # Get the code from GitHub
    github_raw_url = "https://raw.githubusercontent.com/TAG-Research/TAG-Bench/76d5795d6e35f770894d3f180af58b6638964fcf/tag/hand_written.py"
    response = requests.get(github_raw_url)
    code = response.text
    pipeline_functions = re.findall(
        r"(def\s+pipeline_\d+\(\):.+?)(?=\ndef|\Z)", code, re.DOTALL
    )
    token_counts = []
    for f in pipeline_functions:
        function_def, f = f.split("\n", 1)
        pipeline_id = re.search(r"(?<=def pipeline_)\d+", function_def).group()
        if pipeline_id not in use_ids:
            continue
        # Load and remove function definition
        lines = [l.strip() for l in f.split("\n")[1:]]
        lines = [l for l in lines if not l.startswith(("query", "return", "answer"))]
        # lines = [l for l in lines if not "pd.read_csv" in l] # TODO: technically isn't this equivalent to a `... FROM` statement?
        token_counts.append(len(tokenizer.encode("\n".join(lines))))
    return token_counts


def get_blendsql_token_counts(
    dataset: t.List[dict], tokenizer: PreTrainedTokenizerBase
) -> t.List[str]:
    token_counts = []
    for example in dataset:
        if example["BlendSQL"] is None:
            assert example["Query type"] == "Aggregation", print(example)
            continue
        token_counts.append(len(tokenizer.encode(example["BlendSQL"])))
    return token_counts


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    use_ids = set(
        [
            str(example["Query ID"])
            for example in BLENDSQL_ANNOTATED_TAG_DATASET
            if example["Query type"] != "Aggregation"
        ]
    )

    lotus_token_counts = get_lotus_token_counts(use_ids, tokenizer)
    blendsql_token_counts = get_blendsql_token_counts(
        BLENDSQL_ANNOTATED_TAG_DATASET, tokenizer
    )

    print(f"BlendSQL average token count: {np.mean(blendsql_token_counts)}")
    print(f"LOTUS average token count: {np.mean(lotus_token_counts)}")
