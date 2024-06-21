import pytest
import httpx

from blendsql import blend, LLMQA, LLMJoin
from blendsql._smoothie import Smoothie
from blendsql._exceptions import InvalidBlendSQL
from blendsql.db import SQLite
from blendsql.utils import fetch_from_hub
from blendsql.models import OllamaLLM

TEST_OLLAMA_LLM = "qwen:0.5b"


@pytest.fixture(scope="session")
def db() -> SQLite:
    return SQLite(
        fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db")
    )


@pytest.fixture(scope="session")
def ingredients() -> set:
    return {LLMQA, LLMJoin}


@pytest.mark.long
def test_ollama_basic_llmqa(db, ingredients):
    try:
        model = OllamaLLM(TEST_OLLAMA_LLM, caching=False)
        model.model_obj(messages=[{"role": "user", "content": "hello"}])
    except httpx.ConnectError:
        pytest.skip("Ollama server is not running, skipping this test")
    blendsql = """
    {{
        LLMQA(
            'Tell me about this table',
            (SELECT * FROM w LIMIT 5)
        )
    }}
    """
    smoothie = blend(
        query=blendsql, db=db, ingredients=ingredients, default_model=model
    )
    assert not smoothie.df.empty


@pytest.mark.long
def test_ollama_raise_exception(db, ingredients):
    model = OllamaLLM(TEST_OLLAMA_LLM, caching=False)
    with pytest.raises(InvalidBlendSQL):
        _ = blend(
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


@pytest.mark.long
def test_ollama_join(db, ingredients):
    try:
        model = OllamaLLM(TEST_OLLAMA_LLM, caching=False)
        model.model_obj(messages=[{"role": "user", "content": "hello"}])
    except httpx.ConnectError:
        pytest.skip("Ollama server is not running, skipping this test")
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
