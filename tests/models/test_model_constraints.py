import pytest
import pandas as pd

import blendsql
from blendsql.db import Pandas
from blendsql.models import AzurePhiModel


@pytest.fixture(scope="session")
def db() -> Pandas:
    return Pandas(
        {
            "w": pd.DataFrame(
                {
                    "President": [
                        "George Washington",
                        "John Quincy Adams",
                        "Thomas Jefferson",
                        "James Madison",
                        "James Monroe",
                    ],
                    "Party": [
                        "Independent",
                        "Federalist",
                        "Democratic-Republican",
                        "Democratic-Republican",
                        "Democratic-Republican",
                    ],
                    "Key_Events": [
                        "Established federal government, Whiskey Rebellion, farewell address warning against political parties",
                        "XYZ Affair, Alien and Sedition Acts, avoided war with France",
                        "Louisiana Purchase, Lewis and Clark Expedition, ended Barbary Wars",
                        "War of 1812, Hartford Convention, chartered Second Bank of U.S.",
                        "Monroe Doctrine, Missouri Compromise, Era of Good Feelings",
                    ],
                }
            )
        }
    )


def test_alphabet(db, model, ingredients):
    if not isinstance(model, AzurePhiModel):
        pytest.skip()
    blend = lambda query: blendsql.blend(
        query=query,
        default_model=model,
        ingredients=ingredients,
        db=db,
    )
    blendsql_query = """
    SELECT * FROM ( VALUES {{LLMQA('What are the first letters of the alphabet?')}} )
    """
    smoothie = blend(blendsql_query)
    assert "A" in list(smoothie.df.values.flat)

    blendsql_query = """
        SELECT * FROM ( VALUES {{LLMQA('What are the first letters of the alphabet?', modifier="{3}")}} )
        """
    smoothie = blend(blendsql_query)
    assert list(smoothie.df.values.flat) == ["A", "B", "C"]

    blendsql_query = """
    SELECT * FROM ( VALUES {{LLMQA('What are the first letters of the alphabet?', options='α;β;γ;δ', modifier="{3}")}} )
    """
    smoothie = blend(blendsql_query)
    assert list(smoothie.df.values.flat) == ["α", "β", "γ"]

    blendsql_query = """
    SELECT {{
        LLMQA(
            'What is the first letter of the alphabet?',
            options=(SELECT * FROM (VALUES {{LLMQA('List some greek letters')}}))
        )
    }} AS 'response'
    """
    smoothie = blend(blendsql_query)
    assert list(smoothie.df.values.flat)[0].lower() == "alpha"

    blendsql_query = """
    WITH greek_letters AS (
        SELECT * FROM (VALUES {{LLMQA('List some greek letters')}})
    ) SELECT {{
        LLMQA(
            'What is the first letter of the alphabet?', 
            options=(SELECT * FROM greek_letters)
        )}}
    """
    smoothie = blend(blendsql_query)
    assert list(smoothie.df.values.flat)[0].lower() == "alpha"
