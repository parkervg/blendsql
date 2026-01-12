import pytest
import pandas as pd

from blendsql import BlendSQL, config

config.set_async_limit(1)
config.set_deterministic(True)


@pytest.fixture(scope="module")
def bsql() -> BlendSQL:
    return BlendSQL(
        {
            "w": pd.DataFrame(
                {
                    "player_name": ["John Wall", "Jayson Tatum"],
                    "Report": ["He had 2 assists", "He only had 1 assist"],
                    "AnotherOptionalReport": ["He had 26pts", "He scored 51pts!"],
                }
            ),
            "v": pd.DataFrame({"people": ["john", "jayson", "emily"]}),
            "names_and_ages": pd.DataFrame(
                {
                    "Name": ["Tommy", "Sarah", "Tommy"],
                    "Description": ["He is 24 years old", "She's 12", "He's only 3"],
                }
            ),
            "movie_reviews": pd.DataFrame(
                {"review": ["I love this movie!", "This was SO GOOD"]}
            ),
        },
    )


def test_map_to_columns(bsql, model):
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
        """,
        model=model,
    )
    assert smoothie.df.columns.tolist() == ["player_name", "points", "assists"]


def test_map_context_with_duplicate_values(bsql, model):
    smoothie = bsql.execute(
        """
         SELECT *, {{
            LLMMap(
                'How old is {}?',
                Name,
                Description,
                return_type='int'
            )
        }} FROM "names_and_ages"
        """,
        model=model,
    )
    df = smoothie.df
    assert set(df[df["Name"] == "Tommy"]["How old is {}?"].values.tolist()) == {24, 3}
    assert df[df["Description"] == "He's only 3"]["How old is {}?"].values.item() == 3


def test_map_options_to_string(bsql, model):
    """cc00959"""
    smoothie = bsql.execute(
        """
         SELECT * FROM movie_reviews 
         WHERE {{
            LLMMap(
                'What is the sentiment of this review?',
                review,
                options=('POSITIVE', 'NEGATIVE')
            )
        }} = 'POSITIVE'
        """,
        model=model,
    )
    assert not smoothie.df.empty
