import pytest
from tabulate import tabulate
from functools import partial
from blendsql import blend
from blendsql.db import SQLiteDBConnector
from tests.utils import assert_equality, starts_with, get_length, select_first_sorted

tabulate = partial(tabulate, headers="keys", showindex="never")


@pytest.fixture
def db() -> SQLiteDBConnector:
    return SQLiteDBConnector(db_path="./tests/data/single_table.db")


@pytest.fixture
def ingredients() -> set:
    return {starts_with, get_length, select_first_sorted}


def test_simple_exec(db, ingredients):
    blendsql = """
    SELECT * FROM transactions;
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    assert not smoothie.df.empty


def test_nested_exec(db, ingredients):
    blendsql = """
    SELECT DISTINCT description FROM transactions WHERE
        merchant in (
            SELECT merchant FROM transactions
                WHERE amount > 100
        );
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    assert not smoothie.df.empty


def test_error_on_delete(db, ingredients):
    blendsql = """
    DELETE FROM transactions WHERE TRUE;
    """
    with pytest.raises(ValueError):
        _ = blend(
            query=blendsql,
            db=db,
            ingredients=ingredients,
        )


def test_simple_ingredient_exec(db, ingredients):
    blendsql = """
    SELECT * FROM transactions WHERE {{starts_with('Z', 'transactions::merchant')}} = 1;
    """
    sql = """
    SELECT * FROM transactions WHERE merchant LIKE "Z%";
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["Z"])


def test_simple_ingredient_exec_at_end(db, ingredients):
    blendsql = """
    SELECT * FROM transactions WHERE {{starts_with('Z', 'transactions::merchant')}}
    """
    sql = """
    SELECT * FROM transactions WHERE merchant LIKE "Z%";
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["Z"])


def test_simple_ingredient_exec_in_select(db, ingredients):
    blendsql = """
    SELECT {{starts_with('Z', 'transactions::merchant')}} FROM transactions WHERE parent_category = 'Auto & Transport'
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_query(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE parent_category = 'Auto & Transport'
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


def test_nested_ingredient_exec(db, ingredients):
    blendsql = """
    SELECT DISTINCT description FROM transactions WHERE
        merchant in (
            SELECT merchant FROM transactions
                WHERE amount > 100
                AND {{starts_with('Z', 'transactions::merchant')}} = 1
        );
    """
    sql = """
     SELECT DISTINCT description FROM transactions WHERE
        merchant in (
            SELECT merchant FROM transactions
                WHERE amount > 100
                AND merchant LIKE "Z%"
        );
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["Z"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_query(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE amount > 100
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


def test_nonexistent_column_exec(db, ingredients):
    blendsql = """
    SELECT DISTINCT description, merchant, child_category FROM transactions WHERE
       (
           {{starts_with('Z', 'transactions::merchant')}} = 1
           AND child_category = 'this does not exist'
       )
       OR child_category = 'Gifts'
    """
    sql = """
    SELECT DISTINCT description, merchant, child_category FROM transactions WHERE
       (
           merchant LIKE "Z%"
           AND child_category = 'this does not exist'
       )
       OR child_category = 'Gifts'
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["Z"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_query(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE child_category = 'this does not exist'
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


def test_nested_and_exec(db, ingredients):
    blendsql = """
    SELECT DISTINCT description, merchant, child_category FROM transactions WHERE
       (
           {{starts_with('O', 'transactions::merchant')}} = 1
           AND child_category = 'Restaurants & Dining'
       )
       OR child_category = 'Gifts'
    """
    sql = """
    SELECT DISTINCT description, merchant, child_category FROM transactions WHERE
       (
           merchant LIKE "O%"
           AND child_category = 'Restaurants & Dining'
       )
       OR child_category = 'Gifts'
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["O"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_query(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE child_category = 'Restaurants & Dining'
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


def test_multiple_nested_ingredients(db, ingredients):
    blendsql = """
    SELECT DISTINCT description, merchant FROM transactions WHERE
        (
            {{starts_with('P', 'transactions::merchant')}} = 1
            AND {{starts_with('T', 'transactions::description')}} = 1
            AND parent_category = 'Food'
        )
       OR child_category = 'Gifts'
    """
    sql = """
    SELECT DISTINCT description, merchant FROM transactions WHERE
        (
            merchant LIKE "P%"
            AND description LIKE "T%"
            AND parent_category = 'Food'
        )
        OR child_category = 'Gifts'
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["P", "T"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_query(
        """
    SELECT COUNT(DISTINCT merchant) + COUNT(DISTINCT description) FROM transactions WHERE parent_category = 'Food'
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


def test_length_ingredient(db, ingredients):
    blendsql = """
    SELECT {{get_length('length', 'transactions::merchant')}}, merchant
        FROM transactions
        WHERE {{get_length('length', 'transactions::merchant')}} > 1;
    """
    sql = """
    SELECT LENGTH(merchant) as length, merchant
        FROM transactions
        WHERE LENGTH(merchant) > 1;
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_query(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


def test_max_length(db, ingredients):
    blendsql = """
    SELECT MAX({{get_length('length', 'transactions::merchant')}}) as max_length, merchant
        FROM transactions
        WHERE {{get_length('length', 'transactions::merchant')}} > 1;
    """
    sql = """
    SELECT MAX(LENGTH(merchant)) as max_length, merchant
        FROM transactions
        WHERE LENGTH(merchant) > 1;
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_query(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


def test_limit(db, ingredients):
    blendsql = """
    SELECT DISTINCT description, merchant, child_category FROM transactions WHERE
       {{starts_with('P', 'transactions::merchant')}} = 1
       AND child_category = 'Restaurants & Dining'
       LIMIT 1
    """
    sql = """
    SELECT DISTINCT description, merchant, child_category FROM transactions WHERE
        merchant LIKE 'P%'
        AND child_category = 'Restaurants & Dining'
        LIMIT 1
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_query(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE child_category = 'Restaurants & Dining'
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


def test_select(db, ingredients):
    blendsql = """
    SELECT DISTINCT description, merchant, child_category FROM transactions WHERE
       {{select_first_sorted(options='transactions::merchant')}}
    """
    sql = """
    SELECT DISTINCT description, merchant, child_category FROM transactions 
        ORDER BY merchant LIMIT 1
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


def test_ingredient_in_select_stmt(db, ingredients):
    blendsql = """
    SELECT MAX({{get_length('length', 'transactions::merchant')}}) as l FROM transactions
    """
    sql = """
    SELECT MAX(LENGTH(merchant)) as l FROM transactions
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


def test_nested_duplicate_ingredient_calls1(db, ingredients):
    blendsql = """
    SELECT merchant FROM transactions WHERE {{get_length('length', 'transactions::merchant')}} > (SELECT {{get_length('length', 'transactions::merchant')}} FROM transactions WHERE merchant = 'Paypal')
    """
    sql = """
    SELECT merchant FROM transactions WHERE LENGTH(merchant) > (SELECT LENGTH(merchant) FROM transactions WHERE merchant = 'Paypal')
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_query(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


if __name__ == "__main__":
    pytest.main()
