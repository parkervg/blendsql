import pytest
import pandas as pd

from blendsql import BlendSQL, config
from blendsql.ingredients import LLMQA, LLMMap, LLMJoin

config.set_async_limit(1)


@pytest.fixture(scope="module")
def bsql() -> BlendSQL:
    return BlendSQL(
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
        },
        ingredients={LLMQA, LLMMap, LLMJoin},
    )


def test_singers(bsql, model):
    print(model.model_name_or_path)
    _ = bsql.execute(
        """
        SELECT * FROM People p
        WHERE {{LLMMap('Is a famous singer?', 'p::Name')}} = True
        """,
        model=model,
    )

    _ = bsql.execute(
        """
        SELECT * FROM People p
        WHERE p.Name IN {{
            LLMQA('First 3 presidents of the U.S?', modifier='{3}')
        }}
        """,
        model=model,
    )

    _ = bsql.execute(
        query="""
        SELECT GROUP_CONCAT(Name, ', ') AS 'Names',
        {{LLMMap('In which time period did the person live?', 'People::Name', options='Eras::Years')}} AS "Lived During Classification"
        FROM People
        GROUP BY "Lived During Classification"
        """,
        model=model,
    )


def test_alphabet(bsql, constrained_model):
    smoothie = bsql.execute(
        """
        SELECT * FROM ( VALUES {{LLMQA('What are the first letters of the alphabet?', options='A;B;C')}} )
        """,
        model=constrained_model,
    )
    assert "A" in list(smoothie.df.values.flat)

    smoothie = bsql.execute(
        """
            SELECT * FROM ( VALUES {{LLMQA('What are the first capital letters of the alphabet?', options='A;B;C', modifier="{2}")}} )
            """,
        model=constrained_model,
    )
    assert set(smoothie.df.values.flat) == {"A", "B"}

    smoothie = bsql.execute(
        """
            SELECT * FROM ( VALUES {{LLMQA('What are the first letters of the alphabet?', options='α;β;γ', modifier="{3}")}} )
            """,
        model=constrained_model,
    )
    assert set(smoothie.df.values.flat) == {"α", "β", "γ"}

    smoothie = bsql.execute(
        """
            SELECT {{
                LLMQA(
                    'What is the first letter of the alphabet?',
                    options=(SELECT * FROM (VALUES {{LLMQA('List some greek letters', modifier='{1,3}', options='alpha')}})),
                    modifier='{1,3}'
                )
            }} AS 'response'
            """,
        model=constrained_model,
    )
    assert list(smoothie.df.values.flat)[0].lower() == "alpha"

    smoothie = bsql.execute(
        """
            WITH greek_letters AS (
                SELECT * FROM (VALUES {{LLMQA('List some greek letters', modifier='{1,3}', options='alpha')}})
            ) SELECT {{
                LLMQA(
                    'What is the first letter of the alphabet?',
                    options=(SELECT * FROM greek_letters)
                )}}
            """,
        model=constrained_model,
    )
    assert list(smoothie.df.values.flat)[0].lower() == "alpha"
