import typing as t
import numpy as np
import re
import requests
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import seaborn as sns
import matplotlib.pyplot as plt

from tag_queries import BLENDSQL_ANNOTATED_TAG_DATASET


def get_lotus_token_counts(
    use_ids: t.List[str], tokenizer: PreTrainedTokenizerBase
) -> np.ndarray:
    # Get the code from GitHub
    github_raw_url = "https://raw.githubusercontent.com/TAG-Research/TAG-Bench/76d5795d6e35f770894d3f180af58b6638964fcf/tag/hand_written.py"
    response = requests.get(github_raw_url)
    code = response.text
    pipeline_functions = re.findall(
        r"(def\s+pipeline_\d+\(\):.+?)(?=\ndef|\Z)", code, re.DOTALL
    )
    token_counts = []

    def maybe_normalize_csv_path(l: str) -> str:
        """For a fair comparison, we don't want to penalize LOTUS based on where
        they stored the csvs.
        So, "../pandas_dfs/european_football_2/Player.csv" -> "Player.csv"
        """
        if "pd.read_csv" not in l:
            return l
        # Pattern to match the full path and capture just the filename
        pattern = r'pd\.read_csv\(".*?([^/]+\.csv)"\)'
        replacement = r'pd.read_csv("\1")'

        normalized_code = re.sub(pattern, replacement, l)
        return normalized_code

    for f in pipeline_functions:
        function_def, f = f.split("\n", 1)
        pipeline_id = re.search(r"(?<=def pipeline_)\d+", function_def).group()
        if pipeline_id not in use_ids:
            continue
        # Load and remove function definition
        lines = [l.strip() for l in f.split("\n")[1:]]
        lines = [l for l in lines if not l.startswith(("query", "return", "answer"))]
        # lines = [l for l in lines if not "pd.read_csv" in l] # TODO: technically isn't this equivalent to a `... FROM` statement?
        lines = [maybe_normalize_csv_path(l) for l in lines]
        token_counts.append(len(tokenizer.encode("\n".join(lines))))
    return np.array(token_counts)


def get_blendsql_token_counts(
    dataset: t.List[dict], tokenizer: PreTrainedTokenizerBase
) -> np.ndarray:
    token_counts = []
    for example in dataset:
        if example["BlendSQL"] is None:
            assert example["Query type"] == "Aggregation", print(example)
            continue
        token_counts.append(len(tokenizer.encode(example["BlendSQL"])))
    return np.array(token_counts)


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
    assert len(lotus_token_counts) == len(blendsql_token_counts)

    print(f"BlendSQL average token count: {blendsql_token_counts.mean()}")
    print(f"LOTUS average token count: {lotus_token_counts.mean()}")

    # Create a DataFrame for easier plotting
    df = pd.DataFrame(
        {
            "Question": [f"Q{i + 1}" for i in range(len(lotus_token_counts))],
            "LOTUS": lotus_token_counts,
            "BlendSQL": blendsql_token_counts,
        }
    )

    # Convert to long format for seaborn
    df_long = pd.melt(
        df,
        id_vars=["Question"],
        value_vars=["LOTUS", "BlendSQL"],
        var_name="Model",
        value_name="Token Count",
    )

    # Set aesthetic parameters
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    # Create the main bar plot
    ax = sns.barplot(
        x="Question",
        y="Token Count",
        hue="Model",
        data=df_long,
        palette=["#5DA5DA", "#FAA43A"],
        alpha=0.9,
    )

    # Add mean lines
    mean_1 = lotus_token_counts.mean()
    mean_2 = blendsql_token_counts.mean()

    plt.axhline(
        y=mean_1, color="#5DA5DA", linestyle="--", label=f"LOTUS Mean: {mean_1:.1f}"
    )
    plt.axhline(
        y=mean_2, color="#FAA43A", linestyle="--", label=f"BlendSQL Mean: {mean_2:.1f}"
    )

    # Enhance the plot appearance
    plt.title("Program Token Counts on TAG", fontsize=16, pad=20)
    plt.xlabel("Question", fontsize=12)
    plt.ylabel("Number of Tokens", fontsize=12)
    plt.xticks([])

    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(
        handles=handles, labels=labels, title="", loc="upper right", frameon=True
    )

    # Add value labels on top of each bar
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.0f', padding=3)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    # plt.show()
    plt.savefig("token_counts.png", dpi=300, bbox_inches="tight")
