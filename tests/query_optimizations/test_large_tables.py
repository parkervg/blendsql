import pytest
from blendsql import BlendSQL
from blendsql.common.utils import fetch_from_hub


@pytest.fixture(scope="session")
def bsql() -> BlendSQL:
    return BlendSQL(
        fetch_from_hub("european_football_2.sqlite"),
    )


def test_qa_with_map_cascade(bsql, model):
    """351d3be"""
    smoothie = bsql.execute(
        """
    SELECT *
       FROM Player p 
       WHERE player_name = {{
           LLMQA(
               "Say 'Lionel Messi'"
           )
       }} AND {{LLMMap('Is greater than 1?', player_api_id)}} LIMIT 1
    """,
        model=model,
    )
    assert len(smoothie.df) == 1
    assert smoothie.meta.num_generation_calls == 2
    assert smoothie.meta.num_values_passed == 1
    assert smoothie.df.iloc[0]["player_name"] == "Lionel Messi"
