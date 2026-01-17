import pytest
from blendsql import BlendSQL
from blendsql.common.utils import fetch_from_hub


@pytest.fixture(scope="session")
def bsql() -> BlendSQL:
    return BlendSQL(
        fetch_from_hub("european_football_2.sqlite"),
    )


@pytest.mark.timeout(20)
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


@pytest.mark.timeout(20)
def test_qa_with_map_cascade_and_join(bsql, model):
    smoothie = bsql.execute(
        """
        SELECT *
           FROM Player p 
           JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id
           WHERE p.player_name IN {{
               LLMQA(
                   "Who are your top 3 favorite players?",
                   quantifier='{3}'
               )
           }} AND {{LLMMap('Is greater than 1?', p.player_api_id)}} 
        """,
        model=model,
    )
    assert not smoothie.df.empty
    assert smoothie.meta.num_generation_calls <= 4
    assert smoothie.meta.num_values_passed <= 3


# def test_multi_qa_map_cascade(bsql, model):
#     smoothie = bsql.execute(
#         """
#         SELECT *
#            FROM Player p
#            JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id
#            WHERE p.player_name IN {{
#                LLMQA(
#                    "Who are your top 3 favorite players?",
#                    quantifier='{3}'
#                )
#            }} AND {{LLMMap('Is greater than 1?', p.player_api_id)}}
#            AND {{LLMMap('What year were they born?', p.player_name, p.birthday)}} IN (1985, 1987)
#         """, model=model
#     )
#     assert not smoothie.df.empty
#     assert smoothie.meta.num_generation_calls <= 5
#     assert smoothie.meta.num_values_passed <= 5
