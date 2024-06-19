import pytest
import pandas as pd
from blendsql import blend
from blendsql.db import Pandas
from blendsql._exceptions import IngredientException, InvalidBlendSQL
from tests.utils import select_first_option


@pytest.fixture(scope="session")
def db() -> Pandas:
    """Create a dummy db to use in tests."""
    return Pandas(
        pd.DataFrame({"Name": ["Danny", "Emma", "Tony"], "Age": [23, 26, 19]}),
        tablename="w",
    )


def test_error_on_delete1(db):
    blendsql = """
    DELETE FROM w WHERE TRUE;
    """
    with pytest.raises(InvalidBlendSQL):
        _ = blend(
            query=blendsql,
            db=db,
            ingredients=set(),
        )


def test_error_on_delete2(db):
    blendsql = """
    DROP TABLE w;
    """
    with pytest.raises(InvalidBlendSQL):
        _ = blend(
            query=blendsql,
            db=db,
            ingredients=set(),
        )


def test_error_on_invalid_ingredient(db):
    blendsql = """
    SELECT * w WHERE {{ingredient()}} = TRUE
    """
    with pytest.raises(IngredientException):
        _ = blend(
            query=blendsql,
            db=db,
            ingredients={"This is not an ingredient type"},
        )


def test_error_on_bad_options_subquery(db):
    blendsql = """
    SELECT * FROM w 
    WHERE {{
        select_first_option(
            'I am at a nice cafe right now',
            'w::Name',
            options=(SELECT * FROM w)
        )
    }}
    """
    with pytest.raises(InvalidBlendSQL):
        _ = blend(
            query=blendsql,
            db=db,
            ingredients={select_first_option},
        )
