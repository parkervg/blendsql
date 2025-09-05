from datasets import load_dataset
from blendsql import BlendSQL
from blendsql.models import TransformersLLM
from blendsql.ingredients import LLMMap
from blendsql.configure import GLOBAL_HISTORY


if __name__ == "__main__":
    dataset = load_dataset("mrm8488/rotowire-sbnation")
    dataset = dataset["test"].to_pandas()
    dataset["Report"] = dataset["summary"].apply(lambda x: " ".join(x))
    dataset["id"] = dataset.index

    bsql = BlendSQL(
        # dataset,
        {
            "player_name": ["Lebron James", "Kobe Bryant"],
            "Report": ["He scored 42pts and had 2 assists", "1 assist, but 60 points!"],
        },
        model=TransformersLLM("HuggingFaceTB/SmolLM-360M-Instruct", caching=True),
        verbose=True,
        ingredients={LLMMap.from_args(num_few_shot_examples=0)},
    )

    # smoothie = bsql.execute(
    #     """
    #     WITH t AS (
    #         SELECT * FROM w WHERE home_name = {{LLMQA('Which team is from San Antonio?')}}
    #     )
    #     SELECT * FROM t
    #     """
    # )
    # print(smoothie.df)

    # smoothie = bsql.execute(
    #     """
    #     WITH player_list AS (
    #         SELECT {{
    #             LLMMap(
    #                     "Which players that played in this game? Ignore players that are mentioned and did not play. An example output looks like `['Kobe Bryant', 'Jeff Green']`",
    #                     Report,
    #                     return_type='List[str]',
    #                     quantifier='{0,4}'
    #                 )
    #             }} AS players
    #         FROM w LIMIT 2
    #     ) PIVOT (
    #       SELECT
    #         id,
    #         UNNEST(players) AS value,
    #         ROW_NUMBER() OVER (PARTITION BY id ORDER BY (SELECT NULL)) AS position
    #       FROM player_list
    #     )
    #     ON position
    #     USING MAX(value);
    #     """
    # )
    # print(smoothie.df)
    # print()
    # _ = bsql.model.model_obj
    smoothie = bsql.execute(
        """
        WITH player_stats AS (
            SELECT *, {{
            LLMMap(
                'How many points and assists did {} have? Respond in the order [points, assists].', 
                player_name, 
                Report, 
                return_type='List[int]',
                quantifier='{2}'
                )
            }} AS box_score_values
            FROM w
        ) SELECT 
        player_name,
        list_element(box_score_values, 1) AS points,
        list_element(box_score_values, 2) AS assists
        FROM player_stats
        """
    )
    print(smoothie.df)
    print(GLOBAL_HISTORY[-1])

    # smoothie = bsql.execute(
    #     """
    #     WITH joined_context AS (
    #       SELECT *,
    #       'Player: ' || player_name || '\nReport: ' || Report AS context
    #       FROM w
    #     )
    #     SELECT player_name, {{LLMMap('How many assists did the player have?', context, return_type='int')}} FROM joined_context
    #     """
    # )
    # print(smoothie.df)
    # exit()
    # smoothie = bsql.execute(
    #     """
    #     SELECT
    #       ARRAY_AGG(DISTINCT city) AS cities,
    #     FROM w WHERE city = {{LLMQA('Which city is mentioned in Nemo?')}};
    #     """
    # )
    # print(smoothie.df)
    #
    # smoothie = bsql.execute(
    #     """
    #         SELECT {{LLMMap('What was the total score?', score)}} FROM w
    #         WHERE city = 'sydney'
    #     """
    # )
    # print(smoothie.df)
