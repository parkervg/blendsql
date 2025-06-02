import pytest
import pandas as pd

from blendsql import BlendSQL, config
from blendsql.models import LlamaCpp
from .utils import test_starts_with

config.set_async_limit(1)


@pytest.fixture(scope="module")
def bsql() -> BlendSQL:
    return BlendSQL(
        {
            "League": pd.DataFrame(
                [
                    {"id": 1, "country_id": 1, "name": "Z Belgium Jupiler League"},
                    {"id": 1729, "country_id": 1729, "name": "England Premier League"},
                    {"id": 4769, "country_id": 4769, "name": "France Ligue 1"},
                    {"id": 7809, "country_id": 7809, "name": "Germany 1. Bundesliga"},
                    {"id": 10257, "country_id": 10257, "name": "The Italy Serie A"},
                ]
            ),
            "Country": pd.DataFrame(
                [
                    {"id": 1, "name": "Belgium"},
                    {"id": 1729, "name": "England"},
                    {"id": 4769, "name": "France"},
                    {"id": 7809, "name": "Germany"},
                    {"id": 10257, "name": "Italy"},
                ]
            ),
            "Food": pd.DataFrame(
                [
                    {"id": 1, "name": "Broccoli", "calories": 14},
                    {"id": 10257, "name": "Pizza", "calories": 300},
                    {"id": 4769, "name": "Celery", "calories": 1},
                    {"id": 9999, "name": "Wasabi", "calories": 99},
                ]
            ),
        },
        ingredients={test_starts_with},
    )


def test_many_aliases(bsql, model):
    _ = bsql.execute(
        """
        SELECT l.name FROM League l
        JOIN Country c ON l.id = c.id
        WHERE {{LLMMap('Is this country landlocked?', c.name)}} = TRUE
        """,
        model=model,
    )


def test_join_with_duplicate_columns(bsql):
    smoothie = bsql.execute(
        """
        SELECT l.name FROM League l
        JOIN Country c ON l.id = c.id
        WHERE {{test_starts_with('I', c.name)}}
        """,
    )
    assert not smoothie.df.empty


def test_llmqa_options_precedence(bsql, model):
    smoothie = bsql.execute(
        """
        SELECT l.name FROM League l JOIN Country c ON l.id = c.id 
        WHERE l.id < 4769  
        AND l.name = {{LLMQA('Which of these is in Italy?')}}
        """,
        model=model,
    )
    if isinstance(model, LlamaCpp):
        assert smoothie.df.empty


def test_llmjoin_with_alias(bsql, model):
    """1c3e4bf"""
    _ = bsql.execute(
        """
        SELECT l.name, c.name FROM League l
        JOIN Country c ON {{
            LLMJoin(
                l.name, 
                c.name, 
                join_criteria='Align the league to its country'
            )
        }}
        """,
        model=model,
    )
