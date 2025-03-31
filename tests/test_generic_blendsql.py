import pytest
import pandas as pd
from blendsql import BlendSQL
from blendsql.db import Pandas
from blendsql._exceptions import IngredientException, InvalidBlendSQL
from tests.utils import select_first_option


@pytest.fixture(scope="session")
def bsql() -> BlendSQL:
    return BlendSQL(
        db=Pandas(
            pd.DataFrame({"Name": ["Danny", "Emma", "Tony"], "Age": [23, 26, 19]}),
            tablename="w",
        ),
    )


def test_error_on_delete1(bsql):
    with pytest.raises(InvalidBlendSQL):
        _ = bsql.execute(
            query="""
            DELETE FROM w WHERE TRUE;
            """,
        )


def test_error_on_delete2(bsql):
    with pytest.raises(InvalidBlendSQL):
        _ = bsql.execute(
            query="""
            DROP TABLE w;
            """,
        )


def test_error_on_invalid_ingredient(bsql):
    with pytest.raises(IngredientException):
        _ = bsql.execute(
            query="""
            SELECT * w WHERE {{ingredient()}} = TRUE
            """,
            ingredients={"This is not an ingredient type"},
        )


def test_error_on_bad_options_subquery(bsql):
    with pytest.raises(InvalidBlendSQL):
        _ = bsql.execute(
            query="""
            SELECT * FROM w
            WHERE {{
                select_first_option(
                    'I am at a nice cafe right now',
                    'w::Name',
                    options=(SELECT * FROM w)
                )
            }}
            """,
            ingredients={select_first_option},
        )
