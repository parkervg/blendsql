import pytest
from typing import Set

from blendsql import blend, LLMQA, LLMMap, LLMJoin
from blendsql.ingredients import Ingredient
from blendsql._smoothie import Smoothie
from blendsql.db import SQLite
from blendsql.utils import fetch_from_hub
from blendsql.models import TransformersLLM

TEST_TRANSFORMERS_LLM = "Qwen/Qwen1.5-0.5B"


@pytest.fixture(scope="session")
def db() -> SQLite:
    return SQLite(
        fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db")
    )


@pytest.fixture(scope="session")
def ingredients() -> Set[Ingredient]:
    return {LLMQA, LLMMap, LLMJoin}


@pytest.fixture(scope="session")
def model() -> TransformersLLM:
    return TransformersLLM(TEST_TRANSFORMERS_LLM, caching=False)


def test_no_ingredients(db, model, ingredients):
    res = blend(
        query="""
        select * from w
        """,
        db=db,
        blender=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)


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
        blender=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)


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
        blender=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)


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
        blender=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)