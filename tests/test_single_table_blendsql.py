import pytest
from blendsql import blend
from blendsql.db import SQLite, DuckDB
from blendsql.utils import fetch_from_hub
from tests.utils import (
    assert_equality,
    starts_with,
    get_length,
    select_first_sorted,
    get_table_size,
    select_first_option,
)

# Run all below tests on DuckDB and SQLite
sqlite_db = SQLite(fetch_from_hub("single_table.db"))
databases = [
    sqlite_db,
    DuckDB.from_pandas(
        {"transactions": sqlite_db.execute_to_df("SELECT * FROM transactions")}
    ),
]


@pytest.fixture
def dummy_ingredients() -> set:
    return {
        starts_with,
        get_length,
        select_first_sorted,
        get_table_size,
        select_first_option,
    }


@pytest.mark.parametrize("db", databases)
def test_simple_exec(db, dummy_ingredients):
    blendsql = """
    SELECT * FROM transactions;
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    assert not smoothie.df.empty


@pytest.mark.parametrize("db", databases)
def test_nested_exec(db, dummy_ingredients):
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
        ingredients=dummy_ingredients,
    )
    assert not smoothie.df.empty


@pytest.mark.parametrize("db", databases)
def test_simple_ingredient_exec(db, dummy_ingredients):
    blendsql = """
    SELECT * FROM transactions WHERE {{starts_with('Z', 'transactions::merchant')}} = 1;
    """
    sql = """
    SELECT * FROM transactions WHERE merchant LIKE 'Z%';
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["Z"])


@pytest.mark.parametrize("db", databases)
def test_simple_ingredient_exec_at_end(db, dummy_ingredients):
    blendsql = """
    SELECT * FROM transactions WHERE {{starts_with('Z', 'transactions::merchant')}}
    """
    sql = """
    SELECT * FROM transactions WHERE merchant LIKE 'Z%';
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["Z"])


@pytest.mark.parametrize("db", databases)
def test_simple_ingredient_exec_in_select(db, dummy_ingredients):
    blendsql = """
    SELECT {{get_length('Z', 'transactions::merchant')}} FROM transactions WHERE parent_category = 'Auto & Transport' 
    """
    sql = """
    SELECT LENGTH(merchant) FROM transactions WHERE parent_category = 'Auto & Transport'
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["Z"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE parent_category = 'Auto & Transport'
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_nested_ingredient_exec(db, dummy_ingredients):
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
                AND merchant LIKE 'Z%'
        );
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["Z"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE amount > 100
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_nonexistent_column_exec(db, dummy_ingredients):
    """
    NOTE: Converting to CNF would break this one
    since we would get:
        SELECT DISTINCT merchant, child_category FROM transactions WHERE
        (child_category = 'Gifts' OR STRUCT(STRUCT(A())) = 1) AND
        (child_category = 'Gifts' OR child_category = 'this does not exist')
    """
    blendsql = """
    SELECT DISTINCT merchant, child_category FROM transactions WHERE
       (
           {{starts_with('Z', 'transactions::merchant')}} = 1
           AND child_category = 'this does not exist'
       )
       OR child_category = 'Gifts'
       ORDER BY merchant
    """
    sql = """
    SELECT DISTINCT merchant, child_category FROM transactions WHERE
       (
           merchant LIKE 'Z%'
           AND child_category = 'this does not exist'
       )
       OR child_category = 'Gifts'
       ORDER BY merchant
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["Z"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE child_category = 'this does not exist'
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_nested_and_exec(db, dummy_ingredients):
    blendsql = """
    SELECT DISTINCT merchant, child_category FROM transactions WHERE
       (
           {{starts_with('O', 'transactions::merchant')}} = 1
           AND child_category = 'Restaurants & Dining'
       )
       OR child_category = 'Gifts'
       ORDER BY merchant
    """
    sql = """
    SELECT DISTINCT merchant, child_category FROM transactions WHERE
       (
           merchant LIKE 'O%'
           AND child_category = 'Restaurants & Dining'
       )
       OR child_category = 'Gifts'
       ORDER BY merchant
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["O"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE child_category = 'Restaurants & Dining'
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_multiple_nested_ingredients(db, dummy_ingredients):
    blendsql = """
    SELECT DISTINCT child_category, merchant FROM transactions WHERE
        (
            {{starts_with('A', 'transactions::merchant')}} = 1
            AND {{starts_with('T', 'transactions::child_category')}} = 1
            AND parent_category = 'Food'
        )
       OR child_category = 'Gifts'
       ORDER BY merchant
    """
    sql = """
    SELECT DISTINCT child_category, merchant FROM transactions WHERE
        (
            merchant LIKE 'A%'
            AND child_category LIKE 'T%'
            AND parent_category = 'Food'
        )
        OR child_category = 'Gifts'
        ORDER BY merchant
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["A", "T"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) + COUNT(DISTINCT child_category) FROM transactions WHERE parent_category = 'Food'
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_length_ingredient(db, dummy_ingredients):
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
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_max_length(db, dummy_ingredients):
    """In DuckDB, this causes
    `Binder Error: aggregate function calls cannot be nested`
    """
    if isinstance(db, DuckDB):
        pytest.skip("Skipping nested aggregate for DuckDB...")
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
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_limit(db, dummy_ingredients):
    blendsql = """
    SELECT DISTINCT merchant, child_category FROM transactions WHERE
       {{starts_with('P', 'transactions::merchant')}} = 1
       AND child_category = 'Restaurants & Dining'
       ORDER BY merchant
       LIMIT 1
    """
    sql = """
    SELECT DISTINCT merchant, child_category FROM transactions WHERE
        merchant LIKE 'P%'
        AND child_category = 'Restaurants & Dining'
        ORDER BY merchant
        LIMIT 1
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE child_category = 'Restaurants & Dining'
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_select(db, dummy_ingredients):
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
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


@pytest.mark.parametrize("db", databases)
def test_ingredient_in_select_stmt(db, dummy_ingredients):
    blendsql = """
    SELECT MAX({{get_length('length', 'transactions::merchant')}}) as l FROM transactions 
    """
    sql = """
    SELECT MAX(LENGTH(merchant)) as l FROM transactions
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions 
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_ingredient_in_select_stmt_with_filter(db, dummy_ingredients):
    # commit de4a7bc
    blendsql = """
    SELECT MAX({{get_length('length', 'transactions::merchant')}}) as l FROM transactions WHERE child_category = 'Restaurants & Dining'
    """
    sql = """
    SELECT MAX(LENGTH(merchant)) as l FROM transactions WHERE child_category = 'Restaurants & Dining'
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE child_category = 'Restaurants & Dining'
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_nested_duplicate_map_calls(db, dummy_ingredients):
    blendsql = """
    SELECT merchant FROM transactions WHERE {{get_length('length', 'transactions::merchant')}} > (SELECT {{get_length('length', 'transactions::merchant')}} FROM transactions WHERE merchant = 'Paypal')
    """
    sql = """
    SELECT merchant FROM transactions WHERE LENGTH(merchant) > (SELECT LENGTH(merchant) FROM transactions WHERE merchant = 'Paypal')
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_many_duplicate_map_calls(db, dummy_ingredients):
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
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT
    (SELECT COUNT(DISTINCT merchant) FROM transactions WHERE amount > 1300)
    + (SELECT COUNT(DISTINCT cash_flow) FROM transactions WHERE amount > 1300)
    + (SELECT COUNT(DISTINCT child_category) FROM transactions WHERE amount > 1300)
    + (SELECT COUNT(DISTINCT date) FROM transactions WHERE amount > 1300)
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_exists_isolated_qa_call(db, dummy_ingredients):
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
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT (SELECT COUNT(DISTINCT merchant) FROM transactions WHERE amount > 500) + (SELECT COUNT(*) FROM transactions WHERE amount < 500)
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_query_options_arg(db, dummy_ingredients):
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
        ingredients=dummy_ingredients,
    )
    assert len(smoothie.df) == 1
    assert smoothie.df.values.flat[0] == "Paypal"


@pytest.mark.parametrize("db", databases)
def test_apply_limit(db, dummy_ingredients):
    # commit 335c67a
    blendsql = """
    SELECT {{get_length('length', 'transactions::merchant')}} FROM transactions ORDER BY merchant LIMIT 1
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    assert smoothie.meta.num_values_passed == 1


@pytest.mark.parametrize("db", databases)
def test_apply_limit_with_predicate(db, dummy_ingredients):
    # commit 335c67a
    blendsql = """
    SELECT {{get_length('length', 'transactions::merchant')}} 
    FROM transactions 
    WHERE amount > 1300
    ORDER BY merchant LIMIT 3
    """
    sql = """
    SELECT LENGTH(merchant)
    FROM transactions 
    WHERE amount > 1300
    ORDER BY merchant LIMIT 3
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE amount > 1300 LIMIT 3    """
    )[0]
    # We say `<=` here because the ingredient operates over sets, rather than lists
    # So this kind of screws up the `LIMIT` calculation
    # But execution outputs should be the same (tested above)
    assert smoothie.meta.num_values_passed <= passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_where_with_true(db, dummy_ingredients):
    """Makes sure we don't ignore `{column} = TRUE` SQL clauses."""
    blendsql = """
    WITH a AS (
        SELECT LENGTH(merchant) = 5 AS b, merchant FROM transactions 
    )
    SELECT merchant FROM a WHERE a.b = TRUE AND {{starts_with('Z', 'a::merchant')}} 
    """
    sql = """
    WITH a AS (
        SELECT LENGTH(merchant) = 5 AS b, merchant FROM transactions 
    )
    SELECT merchant FROM a WHERE a.b = TRUE AND a.merchant LIKE 'Z%'
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE LENGTH(merchant) = 5
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_where_in_clause(db, dummy_ingredients):
    """Makes sure we don't ignore `{column} = TRUE` SQL clauses."""
    blendsql = """
    SELECT merchant FROM transactions WHERE merchant IN (SELECT merchant FROM transactions WHERE {{starts_with('Z', 'transactions::merchant')}})
    """
    sql = """
    SELECT merchant FROM transactions WHERE merchant IN (SELECT merchant FROM transactions WHERE merchant LIKE 'Z%')
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


@pytest.mark.parametrize("db", databases)
def test_group_by_with_ingredient_alias(db, dummy_ingredients):
    """b28a129"""
    blendsql = """
    SELECT SUM(amount) AS "total amount", 
    {{get_length('transactions::merchant')}} AS "Merchant Length"
    FROM transactions
    GROUP BY "Merchant Length"
    ORDER BY "total amount"
    """
    sql = """
    SELECT SUM(amount) AS "total amount", 
    LENGTH(merchant) AS "Merchant Length"
    FROM transactions
    GROUP BY "Merchant Length"
    ORDER BY "total amount"
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


@pytest.mark.parametrize("db", databases)
def test_null_negation(db, dummy_ingredients):
    """ee3b0c4"""
    blendsql = """
    SELECT merchant FROM transactions
    WHERE merchant IS NOT NULL
    AND {{starts_with('Z', 'transactions::merchant')}}
    """
    sql = """
    SELECT merchant FROM transactions
    WHERE merchant IS NOT NULL
    AND merchant LIKE 'Z%'
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=dummy_ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


if __name__ == "__main__":
    pytest.main()
