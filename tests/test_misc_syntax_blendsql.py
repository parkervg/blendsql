import pytest
import pandas as pd

from blendsql import BlendSQL, config
from blendsql.ingredients import LLMQA, LLMMap, LLMJoin

config.set_async_limit(1)


@pytest.fixture(scope="module")
def bsql() -> BlendSQL:
    return BlendSQL(
        {
            "League": pd.DataFrame(
                [
                    {"id": 1, "country_id": 1, "name": "Belgium Jupiler League"},
                    {"id": 1729, "country_id": 1729, "name": "England Premier League"},
                    {"id": 4769, "country_id": 4769, "name": "France Ligue 1"},
                    {"id": 7809, "country_id": 7809, "name": "Germany 1. Bundesliga"},
                    {"id": 10257, "country_id": 10257, "name": "Italy Serie A"},
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
        },
        ingredients={LLMQA, LLMMap, LLMJoin},
    )


def test_many_aliases(bsql, model):
    _ = bsql.execute(
        """
        SELECT l.name FROM League l
        JOIN Country c ON l.country_id = c.id
        WHERE {{LLMMap('Is this country landlocked?', 'c::name')}} = TRUE
        """,
        model=model,
    )
