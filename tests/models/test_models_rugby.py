import pytest

from blendsql import BlendSQL
from blendsql._smoothie import Smoothie
from blendsql.utils import fetch_from_hub
from blendsql.models import LiteLLM


@pytest.fixture(scope="session")
def bsql() -> BlendSQL:
    return BlendSQL(
        fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db")
    )


@pytest.mark.long
def test_no_ingredients(bsql, model, ingredients):
    res = bsql.execute(
        """
        select * from w
        """,
        model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)


@pytest.mark.long
def test_llmmap(bsql, model, ingredients):
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
        model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
    if isinstance(model, LiteLLM):
        assert set(res.df["venue"].values.tolist()) == {
            "cricket ground",
            "parramatta ground",
        }


@pytest.mark.long
def test_llmjoin(bsql, model, ingredients):
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
        model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
    if isinstance(model, LiteLLM):
        assert res.df["rival"].unique().tolist() == ["nsw waratahs"]


@pytest.mark.long
def test_llmqa(bsql, model, ingredients):
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
        model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
    if isinstance(model, LiteLLM):
        assert res.df["city"].unique().tolist() == ["bathurst"]


@pytest.mark.long
def test_llmmap_with_string(bsql, model, ingredients):
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
        model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
    if isinstance(model, LiteLLM):
        assert res.df["June Count"].values[0] == 6


@pytest.mark.long
def test_unconstrained_llmqa(bsql, model, ingredients):
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
        model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
    if isinstance(model, LiteLLM):
        assert "sports" in res.df.values[0][0].lower()
