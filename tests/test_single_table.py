import pytest
from blendsql import BlendSQL
from blendsql.db import SQLite, DuckDB
from blendsql.common.utils import fetch_from_hub
from tests.utils import (
    assert_equality,
    test_starts_with,
    get_length,
    select_first_sorted,
    get_table_size,
    select_first_option,
)

dummy_ingredients = {
    test_starts_with,
    get_length,
    select_first_sorted,
    get_table_size,
    select_first_option,
}

# Run all below tests on DuckDB and SQLite
bsql_connections = [
    BlendSQL(
        SQLite(fetch_from_hub("single_table.db")),
        ingredients=dummy_ingredients,
    ),
    BlendSQL(
        DuckDB.from_pandas(
            {
                "transactions": SQLite(fetch_from_hub("single_table.db")).execute_to_df(
                    "SELECT * FROM transactions"
                )
            }
        ),
        ingredients=dummy_ingredients,
    ),
]


@pytest.mark.parametrize("bsql", bsql_connections)
def test_simple_exec(bsql):
    smoothie = bsql.execute(
        """
        SELECT * FROM transactions;
        """
    )
    assert not smoothie.df.empty


@pytest.mark.parametrize("bsql", bsql_connections)
def test_nested_exec(bsql):
    smoothie = bsql.execute(
        """
        SELECT DISTINCT merchant FROM transactions WHERE
            merchant in (
                SELECT merchant FROM transactions
                    WHERE amount > 100
            );
        """
    )
    assert not smoothie.df.empty


@pytest.mark.parametrize("bsql", bsql_connections)
def test_simple_ingredient_exec(bsql):
    smoothie = bsql.execute(
        """
        SELECT * FROM transactions WHERE {{test_starts_with('Z', merchant)}} = 1;
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT * FROM transactions WHERE merchant LIKE 'Z%';
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["Z"])


@pytest.mark.parametrize("bsql", bsql_connections)
def test_simple_ingredient_exec_at_end(bsql):
    smoothie = bsql.execute(
        """
        SELECT * FROM transactions WHERE {{test_starts_with('Z', merchant)}}
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT * FROM transactions WHERE merchant LIKE 'Z%';
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["Z"])


@pytest.mark.parametrize("bsql", bsql_connections)
def test_simple_ingredient_exec_in_select(bsql):
    smoothie = bsql.execute(
        """
        SELECT {{get_length(merchant)}} AS "LENGTH(merchant)" FROM transactions WHERE parent_category = 'Auto & Transport'
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT LENGTH(merchant) FROM transactions WHERE parent_category = 'Auto & Transport'
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["Z"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE parent_category = 'Auto & Transport'
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("bsql", bsql_connections)
def test_nested_ingredient_exec(bsql):
    smoothie = bsql.execute(
        """
        SELECT DISTINCT merchant FROM transactions WHERE
            merchant in (
                SELECT merchant FROM transactions
                    WHERE amount > 100
                    AND {{test_starts_with('Z', merchant)}} = 1
            );
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
         SELECT DISTINCT merchant FROM transactions WHERE
            merchant in (
                SELECT merchant FROM transactions
                    WHERE amount > 100
                    AND merchant LIKE 'Z%'
            );
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["Z"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE amount > 100
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("bsql", bsql_connections)
def test_nonexistent_column_exec(bsql):
    """
    NOTE: Converting to CNF would break this one
    since we would get:
        SELECT DISTINCT merchant, child_category FROM transactions WHERE
        (child_category = 'Gifts' OR STRUCT(STRUCT(A())) = 1) AND
        (child_category = 'Gifts' OR child_category = 'this does not exist')
    """
    smoothie = bsql.execute(
        """
        SELECT DISTINCT merchant, child_category FROM transactions WHERE
           (
               {{test_starts_with('Z', merchant)}} = 1
               AND child_category = 'this does not exist'
           )
           OR child_category = 'Gifts'
           ORDER BY merchant
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT DISTINCT merchant, child_category FROM transactions WHERE
           (
               merchant LIKE 'Z%'
               AND child_category = 'this does not exist'
           )
           OR child_category = 'Gifts'
           ORDER BY merchant
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["Z"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE child_category = 'this does not exist'
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("bsql", bsql_connections)
def test_nested_and_exec(bsql):
    smoothie = bsql.execute(
        """
        SELECT DISTINCT merchant, child_category FROM transactions WHERE
           (
               {{test_starts_with('O', merchant)}} = 1
               AND child_category = 'Restaurants & Dining'
           )
           OR child_category = 'Gifts'
           ORDER BY merchant
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT DISTINCT merchant, child_category FROM transactions WHERE
           (
               merchant LIKE 'O%'
               AND child_category = 'Restaurants & Dining'
           )
           OR child_category = 'Gifts'
           ORDER BY merchant
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["O"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE child_category = 'Restaurants & Dining'
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("bsql", bsql_connections)
def test_multiple_nested_ingredients(bsql):
    smoothie = bsql.execute(
        """
        SELECT DISTINCT child_category, merchant FROM transactions WHERE
            (
                {{test_starts_with('A', merchant)}} = 1
                AND {{test_starts_with('T', child_category)}} = 1
                AND parent_category = 'Food'
            )
           OR child_category = 'Gifts'
           ORDER BY merchant
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT DISTINCT child_category, merchant FROM transactions WHERE
            (
                merchant LIKE 'A%'
                AND child_category LIKE 'T%'
                AND parent_category = 'Food'
            )
            OR child_category = 'Gifts'
            ORDER BY merchant
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["A", "T"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) + COUNT(DISTINCT child_category) FROM transactions WHERE parent_category = 'Food'
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("bsql", bsql_connections)
def test_length_ingredient(bsql):
    smoothie = bsql.execute(
        """
        SELECT {{get_length(merchant)}}, merchant
            FROM transactions
            WHERE {{get_length(merchant)}} > 1;
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT LENGTH(merchant) as length, merchant
            FROM transactions
            WHERE LENGTH(merchant) > 1;
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("bsql", bsql_connections)
def test_max_length(bsql):
    """In DuckDB, this causes
    `Binder Error: aggregate function calls cannot be nested`
    """
    if isinstance(bsql.db, DuckDB):
        pytest.skip("Skipping nested aggregate for DuckDB...")
    smoothie = bsql.execute(
        """
        SELECT MAX({{get_length(merchant)}}) as max_length, merchant
            FROM transactions
            WHERE {{get_length(merchant)}} > 1;
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT MAX(LENGTH(merchant)) as max_length, merchant
            FROM transactions
            WHERE LENGTH(merchant) > 1;
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("bsql", bsql_connections)
def test_limit(bsql):
    smoothie = bsql.execute(
        """
        SELECT DISTINCT merchant, child_category FROM transactions WHERE
           {{test_starts_with('P', merchant)}} = 1
           AND child_category = 'Restaurants & Dining'
           ORDER BY merchant
           LIMIT 1
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT DISTINCT merchant, child_category FROM transactions WHERE
            merchant LIKE 'P%'
            AND child_category = 'Restaurants & Dining'
            ORDER BY merchant
            LIMIT 1
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE child_category = 'Restaurants & Dining'
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("bsql", bsql_connections)
def test_select(bsql):
    smoothie = bsql.execute(
        """
        SELECT DISTINCT merchant, child_category FROM transactions WHERE
           merchant = {{select_first_sorted(options=merchant)}}
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT DISTINCT merchant, child_category FROM transactions
            ORDER BY merchant LIMIT 1
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)


@pytest.mark.parametrize("bsql", bsql_connections)
def test_ingredient_in_select_stmt(bsql):
    smoothie = bsql.execute(
        """
        SELECT MAX({{get_length(merchant)}}) as l FROM transactions
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT MAX(LENGTH(merchant)) as l FROM transactions
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("bsql", bsql_connections)
def test_ingredient_in_select_stmt_with_filter(bsql):
    # commit de4a7bc
    smoothie = bsql.execute(
        """
        SELECT MAX({{get_length(merchant)}}) as l FROM transactions WHERE child_category = 'Restaurants & Dining'
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT MAX(LENGTH(merchant)) as l FROM transactions WHERE child_category = 'Restaurants & Dining'
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE child_category = 'Restaurants & Dining'
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("bsql", bsql_connections)
def test_nested_duplicate_map_calls(bsql):
    smoothie = bsql.execute(
        """
        SELECT merchant FROM transactions WHERE {{get_length(merchant)}} > (SELECT {{get_length(merchant)}} FROM transactions WHERE merchant = 'Paypal' LIMIT 1)
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT merchant FROM transactions WHERE LENGTH(merchant) > (SELECT LENGTH(merchant) FROM transactions WHERE merchant = 'Paypal' LIMIT 1)
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("bsql", bsql_connections)
def test_many_duplicate_map_calls(bsql):
    smoothie = bsql.execute(
        """
        SELECT
            {{get_length(merchant)}} AS l1,
            {{get_length(cash_flow)}} AS l2,
            {{get_length(child_category)}} AS l3,
            {{get_length(date)}} AS l4
        FROM transactions WHERE amount > 1300
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT
            LENGTH(merchant) AS l1,
            LENGTH(cash_flow) AS l2,
            LENGTH(child_category) AS l3,
            LENGTH(date) AS l4
        FROM transactions WHERE amount > 1300
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT
    (SELECT COUNT(DISTINCT merchant) FROM transactions WHERE amount > 1300)
    + (SELECT COUNT(DISTINCT cash_flow) FROM transactions WHERE amount > 1300)
    + (SELECT COUNT(DISTINCT child_category) FROM transactions WHERE amount > 1300)
    + (SELECT COUNT(DISTINCT date) FROM transactions WHERE amount > 1300)
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("bsql", bsql_connections)
def test_exists_isolated_qa_call(bsql):
    # commit 7a19e39
    smoothie = bsql.execute(
        """
        SELECT NOT EXISTS (
            SELECT * FROM transactions WHERE {{get_length( merchant)}} > 4 AND amount > 500
        ) OR (
            {{
                get_table_size((select * from transactions where amount < 500))
            }}
        )
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT NOT EXISTS (
            SELECT * FROM transactions WHERE
            LENGTH(merchant) > 4 AND amount > 500
        ) OR (
            select count(*) from transactions where amount < 500
        )
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT (SELECT COUNT(DISTINCT merchant) FROM transactions WHERE amount > 500) + (SELECT COUNT(*) FROM transactions WHERE amount < 500)
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("bsql", bsql_connections)
def test_query_options_arg(bsql):
    # commit 5ffa26d
    smoothie = bsql.execute(
        """
        {{
            select_first_option(
                (SELECT * FROM transactions),
                options=(SELECT DISTINCT merchant FROM transactions WHERE merchant = 'Paypal')
            )
        }}
        """
    )
    assert len(smoothie.df) == 1
    assert smoothie.df.values.flat[0] == "Paypal"


@pytest.mark.parametrize("bsql", bsql_connections)
def test_apply_limit(bsql):
    # commit 335c67a
    smoothie = bsql.execute(
        """
        SELECT {{get_length(merchant)}} FROM transactions ORDER BY merchant LIMIT 1
        """
    )
    assert smoothie.meta.num_values_passed == 1


@pytest.mark.parametrize("bsql", bsql_connections)
def test_apply_limit_with_predicate(bsql):
    # commit 335c67a
    smoothie = bsql.execute(
        """
        SELECT {{get_length(merchant)}}
        FROM transactions
        WHERE amount > 1300
        ORDER BY merchant LIMIT 3
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT LENGTH(merchant)
        FROM transactions
        WHERE amount > 1300
        ORDER BY merchant LIMIT 3
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE amount > 1300 LIMIT 3    """
    )[0]
    # We say `<=` here because the ingredient operates over sets, rather than lists
    # So this kind of screws up the `LIMIT` calculation
    # But execution outputs should be the same (tested above)
    assert smoothie.meta.num_values_passed <= passed_to_ingredient


@pytest.mark.parametrize("bsql", bsql_connections)
def test_where_with_true(bsql):
    """Makes sure we don't ignore `{column} = TRUE` SQL clauses."""
    smoothie = bsql.execute(
        """
        WITH a AS (
            SELECT LENGTH(merchant) = 5 AS b, merchant FROM transactions
        )
        SELECT merchant FROM a WHERE a.b = TRUE AND {{test_starts_with('Z', merchant)}}
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        WITH a AS (
            SELECT LENGTH(merchant) = 5 AS b, merchant FROM transactions
        )
        SELECT merchant FROM a WHERE a.b = TRUE AND a.merchant LIKE 'Z%'
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE LENGTH(merchant) = 5
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("bsql", bsql_connections)
def test_where_in_clause(bsql):
    """Makes sure we don't ignore `{column} = TRUE` SQL clauses."""
    smoothie = bsql.execute(
        """
        SELECT merchant FROM transactions WHERE merchant IN (SELECT merchant FROM transactions WHERE {{test_starts_with('Z', merchant)}})
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT merchant FROM transactions WHERE merchant IN (SELECT merchant FROM transactions WHERE merchant LIKE 'Z%')
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)


@pytest.mark.parametrize("bsql", bsql_connections)
def test_group_by_with_ingredient_alias(bsql):
    """b28a129"""
    smoothie = bsql.execute(
        """
        SELECT SUM(amount) AS "total amount",
        {{get_length(merchant)}} AS "Merchant Length"
        FROM transactions
        GROUP BY "Merchant Length"
        ORDER BY "total amount"
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT SUM(amount) AS "total amount",
        LENGTH(merchant) AS "Merchant Length"
        FROM transactions
        GROUP BY "Merchant Length"
        ORDER BY "total amount"
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)


@pytest.mark.parametrize("bsql", bsql_connections)
def test_null_negation(bsql):
    """ee3b0c4"""
    smoothie = bsql.execute(
        """
        SELECT merchant FROM transactions
        WHERE merchant IS NOT NULL
        AND {{test_starts_with('Z', merchant)}}
        """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT merchant FROM transactions
        WHERE merchant IS NOT NULL
        AND merchant LIKE 'Z%'
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)


@pytest.mark.parametrize("bsql", bsql_connections)
def test_cte_with_ingredient(bsql):
    """c3ec1eb"""
    smoothie = bsql.execute(
        """
        WITH a AS (
            SELECT {{
                get_table_size((select * from transactions where amount < 500))
            }} AS "size"
        ) SELECT {{
            get_table_size(
                (select * from a where a.size > 0)
            )
        }}
        """
    )
    # sql_df = db.execute_to_df(sql)
    # assert_equality(smoothie=smoothie, sql_df=sql_df)


@pytest.mark.parametrize("bsql", bsql_connections)
def test_offset(bsql):
    """SELECT "fruits"."name" FROM fruits OFFSET 2 will break SQLite
    for some reason, DuckDB is ok with it, though.
    """
    smoothie = bsql.execute(
        """
    SELECT merchant FROM transactions t
    WHERE {{test_starts_with('Z', merchant)}} = TRUE
    ORDER BY {{get_length(t.merchant)}} LIMIT 1 OFFSET 2
    """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT merchant FROM transactions t
        WHERE merchant LIKE 'Z%'
        ORDER BY LENGTH(merchant) LIMIT 1 OFFSET 2
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)


@pytest.mark.parametrize("bsql", bsql_connections)
def test_map_in_function(bsql):
    """6cec1a4"""
    smoothie = bsql.execute(
        """
    SELECT merchant FROM transactions t
    WHERE LENGTH(CAST({{get_length(merchant)}} AS TEXT)) = 1
    AND merchant LIKE 'Z%' 
    """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT merchant FROM transactions t
        WHERE LENGTH(CAST(LENGTH(t.merchant) AS TEXT)) = 1
        AND merchant LIKE 'Z%' 
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE merchant LIKE 'Z%' 
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("bsql", bsql_connections)
def test_having(bsql):
    smoothie = bsql.execute(
        """
     SELECT {{get_length(merchant)}} AS l FROM transactions t
     WHERE merchant LIKE 'Z%'
     GROUP BY merchant, l
     HAVING l > 3
     ORDER BY l
    """
    )
    sql_df = bsql.db.execute_to_df(
        """
        SELECT LENGTH(merchant) AS l FROM transactions t
         WHERE merchant LIKE 'Z%'
         GROUP BY merchant 
         HAVING l > 3 
         ORDER BY l
        """
    )
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = bsql.db.execute_to_list(
        """
    SELECT COUNT(DISTINCT merchant) FROM transactions WHERE merchant LIKE 'Z%' 
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


if __name__ == "__main__":
    pytest.main()
