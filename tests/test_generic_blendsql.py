import pytest
from blendsql import blend
from blendsql.db import SQLiteDBConnector
from blendsql.utils import fetch_from_hub
from blendsql.ingredients.ingredient import IngredientException


@pytest.fixture(scope="session")
def db() -> SQLiteDBConnector:
    return SQLiteDBConnector(fetch_from_hub("single_table.db"))


def test_error_on_delete1(db):
    blendsql = """
    DELETE FROM transactions WHERE TRUE;
    """
    with pytest.raises(ValueError):
        _ = blend(
            query=blendsql,
            db=db,
            ingredients=set(),
        )


def test_error_on_delete2(db):
    blendsql = """
    DROP TABLE transactions;
    """
    with pytest.raises(ValueError):
        _ = blend(
            query=blendsql,
            db=db,
            ingredients=set(),
        )


def test_error_on_invalid_ingredient(db):
    blendsql = """
    SELECT * transactions WHERE {{ingredient()}} = TRUE
    """
    with pytest.raises(IngredientException):
        _ = blend(
            query=blendsql,
            db=db,
            ingredients={"This is not an ingredient type"},
        )
