import pytest

from blendsql import blend
from blendsql._smoothie import Smoothie
from blendsql.db import SQLite
from blendsql.utils import fetch_from_hub
from blendsql.models import AnthropicLLM, OpenaiLLM, AzurePhiModel


@pytest.fixture(scope="session")
def db() -> SQLite:
    return SQLite(
        fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db")
    )


@pytest.mark.long
def test_no_ingredients(db, model, ingredients):
    res = blend(
        query="""
        select * from w
        """,
        db=db,
        default_model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)


@pytest.mark.long
def test_llmmap(db, model, ingredients):
    res = blend(
        query="""
        SELECT DISTINCT venue FROM w
          WHERE city = 'sydney' AND {{
              LLMMap(
                  'More than 30 total points?',
                  'w::score'
              )
          }} = TRUE
        """,
        db=db,
        default_model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
    if isinstance(model, (AnthropicLLM, OpenaiLLM)):
        assert set(res.df["venue"].values.tolist()) == {
            "cricket ground",
            "parramatta ground",
        }


@pytest.mark.long
def test_llmjoin(db, model, ingredients):
    res = blend(
        query="""
        SELECT date, rival, score, documents.content AS "Team Description" FROM w
          JOIN {{
              LLMJoin(
                  left_on='documents::title',
                  right_on='w::rival'
              )
          }} WHERE rival = 'nsw waratahs'
        """,
        db=db,
        default_model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
    if isinstance(model, (AnthropicLLM, OpenaiLLM, AzurePhiModel)):
        assert res.df["rival"].unique().tolist() == ["nsw waratahs"]


@pytest.mark.long
def test_llmqa(db, model, ingredients):
    res = blend(
        query="""
        SELECT * FROM w
          WHERE city = {{
              LLMQA(
                  'Which city is located 120 miles west of Sydney?',
                  (SELECT * FROM documents WHERE documents MATCH 'sydney OR 120'),
                  options='w::city'
              )
          }}
        """,
        db=db,
        default_model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
    if isinstance(model, (AnthropicLLM, OpenaiLLM)):
        assert res.df["city"].unique().tolist() == ["bathurst"]


@pytest.mark.long
def test_llmmap_with_string(db, model, ingredients):
    res = blend(
        query="""
        SELECT COUNT(*) AS "June Count" FROM w
        WHERE {{
              LLMMap(
                  "What's the full month name?",
                  'w::date'
              )
          }} = 'June'
        """,
        db=db,
        default_model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
    if isinstance(model, (AnthropicLLM, OpenaiLLM)):
        assert res.df["June Count"].values[0] == 6


@pytest.mark.long
def test_unconstrained_llmqa(db, model, ingredients):
    res = blend(
        query="""
        {{
          LLMQA(
            "What's this table about?",
            (SELECT * FROM w LIMIT 1),
            options='sports;food;politics'
          )
        }}
        """,
        db=db,
        default_model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
    if isinstance(model, (AnthropicLLM, OpenaiLLM, AzurePhiModel)):
        assert "sports" in res.df.values[0][0].lower()
