from datasets import load_dataset
from blendsql import BlendSQL
from blendsql.models import VLLM


if __name__ == "__main__":
    dataset = load_dataset("mrm8488/rotowire-sbnation")
    dataset = dataset["test"].to_pandas()
    # Join the tokenized `Report` to a single string
    dataset["Report"] = dataset["summary"].apply(lambda x: " ".join(x))
    dataset["id"] = dataset.index

    # First, extract player names
    bsql = BlendSQL(
        dataset,
        model=VLLM(
            model_name_or_path="RedHatAI/gemma-3-12b-it-quantized.w4a16",
            base_url="http://127.0.0.1:8000/v1/",
        ),
        verbose=True,
    )

    smoothie = bsql.execute(
        """
        SELECT Report, {{
        LLMMap(
                "Which players mentioned in the text played in this game? Ignore players that are mentioned but did not play.",
                Report,
                return_type='List[str]'
            )
        }} AS players_list FROM w 
        /* Can remove below if you want to run on more data */
        LIMIT 5
        """
    )

    # Load new (Report, player) table to BlendSQL and calculate stats
    expanded_df = (
        smoothie.df()
        .explode("players_list")
        .rename(columns={"players_list": "player"})
        .sort_values(by="player")
        .reset_index(drop=True)
    )

    bsql = BlendSQL(expanded_df, model=model, verbose=True)
    smoothie = bsql.execute(
        """
        WITH player_stats AS (
            SELECT *, {{
                LLMMap(
                    'How many points and assists did {} have? Respond in the order [points, assists]. If a stat is not present for a player, return -1.', 
                    player, 
                    Report,
                    return_type='List[int]',
                    quantifier='{2}'
                )
            }} AS box_score_values
            FROM w
        ) SELECT 
        player,
        Report,
        list_element(box_score_values, 1) AS points,
        list_element(box_score_values, 2) AS assists
        FROM player_stats
        """
    )
    print(smoothie.df())
