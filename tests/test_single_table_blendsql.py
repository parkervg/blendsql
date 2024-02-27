import pytest
from blendsql import blend
from blendsql.db import SQLite
from blendsql.utils import fetch_from_hub
from tests.utils import (
    assert_equality,
    starts_with,
    get_length,
    select_first_sorted,
    get_table_size,
    select_first_option,
)


@pytest.fixture(scope="session")
def db() -> SQLite:
    return SQLite(fetch_from_hub("single_table.db"))


@pytest.fixture
def ingredients() -> set:
    return {
        starts_with,
        get_length,
        select_first_sorted,
        get_table_size,
        select_first_option,
    }


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
    SELECT DISTINCT merchant FROM transactions WHERE
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
    SELECT DISTINCT merchant FROM transactions WHERE
        merchant in (
            SELECT merchant FROM transactions
                WHERE amount > 100
                AND {{starts_with('Z', 'transactions::merchant')}} = 1
        );
    """
    sql = """
     SELECT DISTINCT merchant FROM transactions WHERE
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
    SELECT DISTINCT merchant, child_category FROM transactions WHERE
       (
           {{starts_with('Z', 'transactions::merchant')}} = 1
           AND child_category = 'this does not exist'
       )
       OR child_category = 'Gifts'
    """
    sql = """
    SELECT DISTINCT merchant, child_category FROM transactions WHERE
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
    SELECT DISTINCT merchant, child_category FROM transactions WHERE
       (
           {{starts_with('O', 'transactions::merchant')}} = 1
           AND child_category = 'Restaurants & Dining'
       )
       OR child_category = 'Gifts'
    """
    sql = """
    SELECT DISTINCT merchant, child_category FROM transactions WHERE
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
    SELECT DISTINCT child_category, merchant FROM transactions WHERE
        (
            {{starts_with('A', 'transactions::merchant')}} = 1
            AND {{starts_with('T', 'transactions::child_category')}} = 1
            AND parent_category = 'Food'
        )
       OR child_category = 'Gifts'
    """
    sql = """
    SELECT DISTINCT child_category, merchant FROM transactions WHERE
        (
            merchant LIKE 'A%'
            AND child_category LIKE 'T%'
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
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["A", "T"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_query(
        """
    SELECT COUNT(DISTINCT merchant) + COUNT(DISTINCT child_category) FROM transactions WHERE parent_category = 'Food'
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
    SELECT DISTINCT merchant, child_category FROM transactions WHERE
       {{starts_with('P', 'transactions::merchant')}} = 1
       AND child_category = 'Restaurants & Dining'
       LIMIT 1
    """
    sql = """
    SELECT DISTINCT merchant, child_category FROM transactions WHERE
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
    SELECT DISTINCT merchant, child_category FROM transactions WHERE
       merchant = {{select_first_sorted(options='transactions::merchant')}}
    """
    sql = """
    SELECT DISTINCT merchant, child_category FROM transactions 
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


def test_nested_duplicate_map_calls(db, ingredients):
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


def test_many_duplicate_map_calls(db, ingredients):
    blendsql = """
    SELECT 
        {{get_length('length', 'transactions::merchant')}} AS l1,
        {{get_length('length', 'transactions::cash_flow')}} AS l2,
        {{get_length('length', 'transactions::child_category')}} AS l3,
        {{get_length('length', 'transactions::date')}} AS l4 
    FROM transactions WHERE amount > 1300
    """
    sql = """
    SELECT 
        LENGTH(merchant) AS l1,
        LENGTH(cash_flow) AS l2,
        LENGTH(child_category) AS l3,
        LENGTH(date) AS l4
    FROM transactions WHERE amount > 1300    
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
    SELECT
    (SELECT COUNT(DISTINCT merchant) FROM transactions WHERE amount > 1300)
    + (SELECT COUNT(DISTINCT cash_flow) FROM transactions WHERE amount > 1300)
    + (SELECT COUNT(DISTINCT child_category) FROM transactions WHERE amount > 1300)
    + (SELECT COUNT(DISTINCT date) FROM transactions WHERE amount > 1300)
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


def test_exists_isolated_qa_call(db, ingredients):
    # commit 7a19e39
    blendsql = """
    SELECT NOT EXISTS (
        SELECT * FROM transactions WHERE {{get_length('length', 'transactions::merchant')}} > 4 AND amount > 500
    ) OR (
        {{
            get_table_size('Table size?', (select * from transactions where amount < 500))
        }}
    )
    """
    sql = """
    SELECT NOT EXISTS (
        SELECT * FROM transactions WHERE 
        LENGTH(merchant) > 4 AND amount > 500
    ) OR (
        select count(*) from transactions where amount < 500
    )   
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
    SELECT (SELECT COUNT(DISTINCT merchant) FROM transactions WHERE amount > 500) + (SELECT COUNT(*) FROM transactions WHERE amount < 500)
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


def test_query_options_arg(db, ingredients):
    # commit 5ffa26d
    blendsql = """
    {{
        select_first_option(
            'I hope this test works',
            (SELECT * FROM transactions),
            options=(SELECT DISTINCT merchant FROM transactions WHERE merchant = 'Paypal')
        )
    }}
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    assert len(smoothie.df) == 1
    assert smoothie.df.values.flat[0] == "Paypal"


if __name__ == "__main__":
    pytest.main()
