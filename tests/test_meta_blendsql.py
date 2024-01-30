import pytest
from tabulate import tabulate
from functools import partial
from blendsql import blend
from blendsql.db import SQLiteDBConnector
from pathlib import Path
from tests.utils import starts_with, get_length

tabulate = partial(tabulate, headers="keys", showindex="never")


@pytest.fixture
def db() -> SQLiteDBConnector:
    return SQLiteDBConnector(db_path="./tests/data/single_table.db")


@pytest.fixture
def ingredients() -> set:
    return {starts_with, get_length}


def test_save_recipe(db, ingredients):
    blendsql = """
    SELECT * FROM transactions WHERE {{starts_with('Z', 'transactions::merchant')}}
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    assert not smoothie.df.empty
    save_path = "test_saved_recipe"
    filenames = [save_path + ext for ext in [".ipynb", ".html"]]
    smoothie.save_recipe(save_path + ".ipynb", "My Test", as_html=True)
    for filename in filenames:
        p = Path(filename)
        assert p.exists()
        p.unlink()
