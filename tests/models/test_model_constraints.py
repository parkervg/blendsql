import pytest
import pandas as pd

import blendsql
from blendsql.models import ConstrainedModel
from blendsql._smoothie import Smoothie
from blendsql.db import Pandas

blendsql.config.set_async_limit(1)


@pytest.fixture(scope="session")
def db() -> Pandas:
    return Pandas(
        {
            "People": pd.DataFrame(
                {
                    "Name": [
                        "George Washington",
                        "John Quincy Adams",
                        "Thomas Jefferson",
                        "James Madison",
                        "James Monroe",
                        "Alexander Hamilton",
                        "Sabrina Carpenter",
                        "Charli XCX",
                        "Elon Musk",
                        "Michelle Obama",
                        "Elvis Presley",
                    ],
                    "Known_For": [
                        "Established federal government, First U.S. President",
                        "XYZ Affair, Alien and Sedition Acts",
                        "Louisiana Purchase, Declaration of Independence",
                        "War of 1812, Constitution",
                        "Monroe Doctrine, Missouri Compromise",
                        "Created national bank, Federalist Papers",
                        "Nonsense, Emails I Cant Send, Mean Girls musical",
                        "Crash, How Im Feeling Now, Boom Clap",
                        "Tesla, SpaceX, Twitter/X acquisition",
                        "Lets Move campaign, Becoming memoir",
                        "14 Grammys, King of Rock n Roll",
                    ],
                }
            ),
            "Eras": pd.DataFrame({"Years": ["1800-1900", "1900-2000", "2000-Now"]}),
        }
    )


def test_singers(db, model, ingredients):
    if isinstance(model, ConstrainedModel):
        pytest.skip()
    res = blendsql.blend(
        query="""
        SELECT * FROM People p
        WHERE {{LLMMap('Is a singer?', 'p::Name')}} = True
        """,
        default_model=model,
        ingredients=ingredients,
        db=db,
    )
    assert set(res.df["Name"].tolist()) == {
        "Sabrina Carpenter",
        "Charli XCX",
        "Elvis Presley",
    }
    res = blendsql.blend(
        query="""
        SELECT GROUP_CONCAT(Name, ', ') AS 'Names',
        {{LLMMap('In which time period did the person live?', 'People::Name', options='Eras::Years')}} AS "Lived During Classification"
        FROM People
        GROUP BY "Lived During Classification"
        """,
        default_model=model,
        ingredients=ingredients,
        db=db,
    )
    assert isinstance(res, Smoothie)


def test_alphabet(db, model, ingredients):
    if not isinstance(model, ConstrainedModel):
        pytest.skip()
    blend = lambda query: blendsql.blend(
        query=query,
        default_model=model,
        ingredients=ingredients,
        db=db,
    )
    blendsql_query = """
    SELECT * FROM ( VALUES {{LLMQA('What are the first letters of the alphabet?', options='A;B;C')}} )
    """
    smoothie = blend(blendsql_query)
    assert "A" in list(smoothie.df.values.flat)

    blendsql_query = """
        SELECT * FROM ( VALUES {{LLMQA('What are the first capital letters of the alphabet?', options='A;B;C', modifier="{2}")}} )
        """
    smoothie = blend(blendsql_query)
    assert set(smoothie.df.values.flat) == {"A", "B"}

    blendsql_query = """
    SELECT * FROM ( VALUES {{LLMQA('What are the first letters of the alphabet?', options='α;β;γ', modifier="{3}")}} )
    """
    smoothie = blend(blendsql_query)
    assert set(smoothie.df.values.flat) == {"α", "β", "γ"}

    blendsql_query = """
    SELECT {{
        LLMQA(
            'What is the first letter of the alphabet?',
            options=(SELECT * FROM (VALUES {{LLMQA('List some greek letters', modifier='{1,3}', options='alpha')}})),
            modifier='{1,3}'
        )
    }} AS 'response'
    """
    smoothie = blend(blendsql_query)
    assert list(smoothie.df.values.flat)[0].lower() == "alpha"

    blendsql_query = """
    WITH greek_letters AS (
        SELECT * FROM (VALUES {{LLMQA('List some greek letters', modifier='{1,3}', options='alpha')}})
    ) SELECT {{
        LLMQA(
            'What is the first letter of the alphabet?',
            options=(SELECT * FROM greek_letters)
        )}}
    """
    smoothie = blend(blendsql_query)
    assert list(smoothie.df.values.flat)[0].lower() == "alpha"
