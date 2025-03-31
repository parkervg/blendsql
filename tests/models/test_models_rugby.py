import pytest
from guidance.chat import Phi3MiniChatTemplate, ChatMLTemplate

from blendsql import BlendSQL
from blendsql._smoothie import Smoothie
from blendsql.utils import fetch_from_hub
from blendsql.models import LiteLLM, TransformersLLM


@pytest.fixture(scope="session")
def bsql() -> BlendSQL:
    return BlendSQL(
        fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db")
    )

@pytest.fixture(scope="session")
def constrained_model():
    model = TransformersLLM(
        "meta-llama/Llama-3.2-3B-Instruct",
        config={"chat_template": Phi3MiniChatTemplate, "device_map": "auto"},
        caching=False
    )
    yield model

@pytest.mark.long
def test_no_ingredients(bsql, constrained_model, ingredients):
    res = bsql.execute(
        """
        select * from w
        """,
        model=constrained_model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)


@pytest.mark.long
def test_llmmap(bsql, constrained_model, ingredients):
    res = bsql.execute(
        """
        SELECT DISTINCT venue FROM w
          WHERE city = 'sydney' AND {{
              LLMMap(
                  'More than 30 total points?',
                  'w::score'
              )
          }} = TRUE
        """,
        model=constrained_model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
    if isinstance(constrained_model, LiteLLM):
        assert set(res.df["venue"].values.tolist()) == {
            "cricket ground",
            "parramatta ground",
        }


@pytest.mark.long
def test_llmjoin(bsql, constrained_model, ingredients):
    res = bsql.execute(
        """
        SELECT date, rival, score, documents.content AS "Team Description" FROM w
          JOIN {{
              LLMJoin(
                  left_on='documents::title',
                  right_on='w::rival'
              )
          }} WHERE rival = 'nsw waratahs'
        """,
        model=constrained_model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
    if isinstance(constrained_model, LiteLLM):
        assert res.df["rival"].unique().tolist() == ["nsw waratahs"]


@pytest.mark.long
def test_llmqa(bsql, constrained_model, ingredients):
    res = bsql.execute(
        """
        SELECT * FROM w
          WHERE city = {{
              LLMQA(
                  'Which city is located 120 miles west of Sydney?',
                  (SELECT * FROM documents WHERE documents MATCH 'sydney OR 120' LIMIT 2),
                  options='w::city'
              )
          }}
        """,
        model=constrained_model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
    if isinstance(constrained_model, LiteLLM):
        assert res.df["city"].unique().tolist() == ["bathurst"]


@pytest.mark.long
def test_llmmap_with_string(bsql, constrained_model, ingredients):
    res = bsql.execute(
        """
        SELECT COUNT(*) AS "June Count" FROM w
        WHERE {{
              LLMMap(
                  "What's the full month name?",
                  'w::date'
              )
          }} = 'June'
        """,
        model=constrained_model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
    if isinstance(constrained_model, LiteLLM):
        assert res.df["June Count"].values[0] == 6


@pytest.mark.long
def test_unconstrained_llmqa(bsql, constrained_model, ingredients):
    res = bsql.execute(
        """
        {{
          LLMQA(
            "What's this table about?",
            (SELECT * FROM w LIMIT 1),
            options='sports;food;politics'
          )
        }}
        """,
        model=constrained_model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
    if isinstance(constrained_model, LiteLLM):
        assert "sports" in res.df.values[0][0].lower()
