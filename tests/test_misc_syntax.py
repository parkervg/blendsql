import pytest
import pandas as pd

from blendsql import BlendSQL, config, GLOBAL_HISTORY
from blendsql.db import DuckDB
from .utils import test_starts_with

config.set_async_limit(1)
config.set_deterministic(True)


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
    _ = bsql.execute(
        """
        SELECT l.name FROM League l JOIN Country c ON l.id = c.id 
        WHERE l.id < 4769  
        AND l.name = {{LLMQA('Which of these has the word ''Italy''?')}}
        """,
        model=model,
    )
    assert (
        "['Z Belgium Jupiler League', 'England Premier League', 'France Ligue 1', 'Germany 1. Bundesliga', 'The Italy Serie A']"
        in GLOBAL_HISTORY[-1]
    )


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


def test_llmmap_with_multi_ctes(bsql, model):
    """ae65c2b"""
    smoothie = bsql.execute(
        """
        WITH t1 AS (
            SELECT *,
            {{LLMMap('Is this in France?', name)}}
            FROM League
        ), t2 AS (
            SELECT * FROM t1 
        ) SELECT * FROM t2 WHERE id > 5000
        """,
        model=model,
    )
    assert (
        smoothie.meta.num_values_passed
        == bsql.db.execute_to_list(
            "SELECT COUNT(DISTINCT name) FROM League", to_type=int
        )[0]
    )


def test_duckdb_read_text(bsql, model):
    if not isinstance(bsql.db, DuckDB):
        pytest.skip()
    _ = bsql.execute(
        """
            SELECT {{
            LLMQA(
                'Describe BlendSQL in 50 words.',
                context=(
                    SELECT content[0:5000] AS "README"
                    FROM read_text('https://raw.githubusercontent.com/parkervg/blendsql/main/README.md')
                )
            )
        }} AS answer
        """,
        model=model,
    )


def test_options_with_return_type(bsql, model):
    """327a17c"""
    smoothie = bsql.execute(
        """
            SELECT * FROM VALUES {{
                LLMQA(
                    'List some countries.', 
                    return_type='List[str]', 
                    options=(SELECT name FROM Country),
                    quantifier='{2}'
                )
            }}
        """,
        model=model,
    )
    assert len(smoothie.df.columns) == 2
