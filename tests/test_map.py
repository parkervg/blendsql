import pytest
import pandas as pd

from blendsql import BlendSQL, config
from blendsql.models import ConstrainedModel

config.set_async_limit(1)


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
        },
    )


def test_map_to_columns(bsql, model):
    if not isinstance(model, ConstrainedModel):
        pytest.skip()
    smoothie = bsql.execute(
        """
        WITH player_stats AS (
            SELECT *, {{
            LLMMap(
                'How many points and assists did {} have? Respond in the order [points, assists].', 
                player_name, 
                Report, 
                AnotherOptionalReport, 
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
