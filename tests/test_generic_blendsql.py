import pytest
import pandas as pd
import sqlite3
from pathlib import Path
from blendsql import blend
from blendsql.db import SQLite
from blendsql.ingredients.ingredient import IngredientException


@pytest.fixture(scope="session")
def db() -> SQLite:
    """Create a dummy sqlite db to use in tests."""
    dbpath = "./test_generic.db"
    df = pd.DataFrame({"Name": ["Danny", "Emma", "Tony"], "Age": [23, 26, 19]})
    con = sqlite3.connect(dbpath)
    df.to_sql("w", con=con)
    con.close()
    yield SQLite(dbpath)
    Path(dbpath).unlink()


def test_error_on_delete1(db):
    blendsql = """
    DELETE FROM w WHERE TRUE;
    """
    with pytest.raises(ValueError):
        _ = blend(
            query=blendsql,
            db=db,
            ingredients=set(),
        )


def test_error_on_delete2(db):
    blendsql = """
    DROP TABLE w;
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
