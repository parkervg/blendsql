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
    blendsql_query = """
    SELECT * FROM ( VALUES {{LLMQA('What are the first letters of the alphabet?')}} )
    """
    smoothie = blendsql.blend(
        query=blendsql_query,
        default_model=model,
        ingredients=ingredients,
        db=db,
    )
    assert "A" in smoothie.df.values.tolist()[0]

    blendsql_query = """
        SELECT * FROM ( VALUES {{LLMQA('What are the first letters of the alphabet?', modifier="{3}")}} )
        """
    smoothie = blendsql.blend(
        query=blendsql_query,
        default_model=model,
        ingredients=ingredients,
        db=db,
    )
    assert smoothie.df.values.tolist()[0] == ["A", "B", "C"]

    blendsql_query = """
    SELECT * FROM ( VALUES {{LLMQA('What are the first letters of the alphabet?', options='α;β;γ;δ', modifier="{3}")}} )
    """
    smoothie = blendsql.blend(
        query=blendsql_query,
        default_model=model,
        ingredients=ingredients,
        db=db,
    )
    assert smoothie.df.values.tolist()[0] == ["α", "β", "γ"]
