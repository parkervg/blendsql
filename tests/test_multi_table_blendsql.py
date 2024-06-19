import pytest
from blendsql import blend
from blendsql.db import SQLite
from blendsql.utils import fetch_from_hub
from tests.utils import (
    assert_equality,
    starts_with,
    get_length,
    select_first_sorted,
    do_join,
    return_aapl,
    get_table_size,
    select_first_option,
)

databases = [
    SQLite(fetch_from_hub("multi_table.db")),
    # DuckDB.from_sqlite(fetch_from_hub("multi_table.db")),
]


@pytest.fixture
def ingredients() -> set:
    return {
        starts_with,
        get_length,
        select_first_sorted,
        # Disable skrub_joiner on JoinIngredient
        # This way, we receive all values
        do_join.from_args(use_skrub_joiner=False),
        return_aapl,
        get_table_size,
        select_first_option,
    }


@pytest.mark.parametrize("db", databases)
def test_simple_multi_exec(db, ingredients):
    """Test with multiple tables.
    Also ensures we only pass what is neccessary to the external ingredient F().
    "Show me the price of tech stocks in my portfolio that start with 'A'"
    """
    blendsql = """
    SELECT Symbol, "North America", "Japan" FROM geographic
        WHERE geographic.Symbol IN (
            SELECT Symbol FROM portfolio
            WHERE {{starts_with('A', 'portfolio::Description')}} = 1
            AND portfolio.Symbol in (
                SELECT Symbol FROM constituents
                WHERE constituents.Sector = 'Information Technology'
            )
        )
    """
    sql = """
    SELECT Symbol, "North America", "Japan" FROM geographic
        WHERE geographic.Symbol IN (
            SELECT Symbol FROM portfolio
            WHERE portfolio.Description LIKE 'A%'
            AND Symbol in (
                SELECT Symbol FROM constituents
                WHERE constituents.Sector = 'Information Technology'
            )
        )
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["A"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT Symbol) FROM portfolio WHERE Symbol in
    (
        SELECT Symbol FROM constituents
                WHERE sector = 'Information Technology'
    )
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_join_multi_exec(db, ingredients):
    blendsql = """
    SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
        FROM account_history
        LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
        WHERE constituents.Sector = 'Information Technology'
        AND {{starts_with('A', 'constituents::Name')}} = 1
        AND lower(account_history.Action) like '%dividend%'
        ORDER BY "Total Dividend Payout ($$)"
        """
    sql = """
    SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
        FROM account_history
        LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
        WHERE constituents.Sector = 'Information Technology'
        AND constituents.Name LIKE 'A%'
        AND lower(account_history.Action) like '%dividend%'
        ORDER BY "Total Dividend Payout ($$)"
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["A"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT Name) FROM constituents WHERE Sector = 'Information Technology'
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_join_not_qualified_multi_exec(db, ingredients):
    """Same test as test_join_multi_exec(), but without qualifying columns if we don't need to.
    i.e. 'Action' and 'Sector' don't have tablename preceding them.
    commit fefbc0a
    """
    blendsql = """
    SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
        FROM account_history
        LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
        WHERE Sector = 'Information Technology'
        AND {{starts_with('A', 'constituents::Name')}} = 1
        AND lower(Action) like '%dividend%'
        ORDER BY "Run Date"
        """
    sql = """
    SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
        FROM account_history
        LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
        WHERE Sector = 'Information Technology'
        AND constituents.Name LIKE 'A%'
        AND lower(Action) like '%dividend%'
        ORDER BY "Run Date"
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["A"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT Name) FROM constituents WHERE Sector = 'Information Technology'
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_select_multi_exec(db, ingredients):
    blendsql = """
    SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
        FROM account_history
        LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
        WHERE constituents.Sector = {{select_first_sorted(options='constituents::Sector')}}
        AND lower(account_history.Action) like '%dividend%'
    """
    sql = """
    SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
        FROM account_history
        LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
        WHERE constituents.Sector = (
            SELECT Sector FROM constituents
            ORDER BY Sector LIMIT 1
        )
        AND lower(account_history.Action) like '%dividend%'
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


@pytest.mark.parametrize("db", databases)
def test_complex_multi_exec(db, ingredients):
    """
    Below yields a tie in constituents.Name lengths, with 'Amgen' and 'Cisco'.
    DuckDB has different sorting behavior depending on the subset that's passed?
    """
    blendsql = """
    SELECT DISTINCT Name FROM constituents
    LEFT JOIN account_history ON constituents.Symbol = account_history.Symbol
    LEFT JOIN portfolio on constituents.Symbol = portfolio.Symbol
    WHERE account_history."Run Date" > '2021-02-23'
    AND ({{get_length('n_length', 'constituents::Name')}} > 3 OR {{starts_with('A', 'portfolio::Symbol')}})
    AND portfolio.Symbol IS NOT NULL
    ORDER BY LENGTH(constituents.Name), constituents.Name LIMIT 4
    """
    sql = """
    SELECT DISTINCT Name FROM constituents
    LEFT JOIN account_history ON constituents.Symbol = account_history.Symbol
    LEFT JOIN portfolio on constituents.Symbol = portfolio.Symbol
    WHERE account_history."Run Date" > '2021-02-23'
    AND (LENGTH(constituents.Name) > 3 OR portfolio.Symbol LIKE 'A%')
    AND portfolio.Symbol IS NOT NULL
    ORDER BY LENGTH(constituents.Name), constituents.Name LIMIT 4
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


@pytest.mark.parametrize("db", databases)
def test_complex_not_qualified_multi_exec(db, ingredients):
    """Same test as test_complex_multi_exec(), but without qualifying columns if we don't need to.
    commit fefbc0a
    """
    blendsql = """
    SELECT DISTINCT constituents.Symbol, Action FROM constituents
    LEFT JOIN account_history ON constituents.Symbol = account_history.Symbol
    LEFT JOIN portfolio on constituents.Symbol = portfolio.Symbol
    WHERE "Run Date" > '2021-02-23'
    AND ({{get_length('n_length', 'constituents::Name')}} > 3 OR {{starts_with('A', 'portfolio::Symbol')}})
    AND portfolio.Symbol IS NOT NULL
    ORDER BY constituents.Symbol, LENGTH(Action) LIMIT 4
    """
    sql = """
    SELECT DISTINCT constituents.Symbol, Action FROM constituents
    LEFT JOIN account_history ON constituents.Symbol = account_history.Symbol
    LEFT JOIN portfolio on constituents.Symbol = portfolio.Symbol
    WHERE "Run Date" > '2021-02-23'
    AND (LENGTH(constituents.Name) > 3 OR portfolio.Symbol LIKE 'A%')
    AND portfolio.Symbol IS NOT NULL
    ORDER BY constituents.Symbol, LENGTH(Action) LIMIT 4
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


@pytest.mark.parametrize("db", databases)
def test_join_ingredient_multi_exec(db, ingredients):
    blendsql = """
    SELECT Account, Quantity FROM returns
    JOIN {{
        do_join(
            left_on='account_history::Account',
            right_on='returns::Annualized Returns'
        )
    }}
    """
    sql = """
    SELECT Account, Quantity FROM returns JOIN account_history ON returns."Annualized Returns" = account_history."Account"
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


@pytest.mark.parametrize("db", databases)
def test_qa_equals_multi_exec(db, ingredients):
    blendsql = """
    SELECT Action FROM account_history
    WHERE Symbol = {{return_aapl()}}
    """
    sql = """
    SELECT Action FROM account_history
    WHERE Symbol = 'AAPL'
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


@pytest.mark.parametrize("db", databases)
def test_table_alias_multi_exec(db, ingredients):
    blendsql = """
    SELECT Symbol FROM portfolio AS w
        WHERE {{starts_with('A', 'w::Symbol')}} = TRUE
        AND LENGTH(w.Symbol) > 3
    """
    sql = """
    SELECT Symbol FROM portfolio AS w
        WHERE w.Symbol LIKE 'A%'
        AND LENGTH(w.Symbol) > 3
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["A"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT Symbol) FROM portfolio WHERE LENGTH(Symbol) > 3
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_subquery_alias_multi_exec(db, ingredients):
    blendsql = """
    SELECT Symbol FROM (
        SELECT DISTINCT Symbol FROM portfolio WHERE Symbol IN (
            SELECT Symbol FROM portfolio WHERE Quantity > 200
        )
    ) AS w WHERE {{starts_with('F', 'w::Symbol')}} = TRUE AND LENGTH(w.Symbol) > 3
    """
    sql = """
    SELECT Symbol FROM (
        SELECT DISTINCT Symbol FROM portfolio WHERE Symbol IN (
            SELECT Symbol FROM portfolio WHERE Quantity > 200
        )
    ) AS w WHERE w.Symbol LIKE 'F%' AND LENGTH(w.Symbol) > 3
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["F"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT Symbol) FROM portfolio WHERE LENGTH(Symbol) > 3 AND Quantity > 200
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_cte_qa_multi_exec(db, ingredients):
    blendsql = """
   {{
        get_table_size(
            (
                WITH a AS (
                    SELECT * FROM (SELECT DISTINCT * FROM portfolio) as w
                        WHERE {{starts_with('F', 'w::Symbol')}} = TRUE
                ) SELECT * FROM a WHERE LENGTH(a.Symbol) > 2
            )
        )
    }}
    """
    sql = """
    WITH a AS (
        SELECT * FROM (SELECT DISTINCT * FROM portfolio) as w
            WHERE w.Symbol LIKE 'F%'
    ) SELECT COUNT(*) FROM a WHERE LENGTH(a.Symbol) > 2
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["F"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_map_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT Symbol) FROM portfolio
    """
    )[0]
    # We also need to factor in what we passed to QA ingredient
    passed_to_qa_ingredient = db.execute_to_list(
        """
    WITH a AS (
        SELECT * FROM (SELECT DISTINCT * FROM portfolio) as w
            WHERE w.Symbol LIKE 'F%'
    ) SELECT COUNT(*) FROM a WHERE LENGTH(a.Symbol) > 2
    """
    )[0]
    assert (
        smoothie.meta.num_values_passed
        == passed_to_qa_ingredient + passed_to_map_ingredient
    )


@pytest.mark.parametrize("db", databases)
def test_cte_qa_named_multi_exec(db, ingredients):
    blendsql = """
   {{
        get_table_size(
            (
                WITH a AS (
                    SELECT * FROM (SELECT DISTINCT * FROM portfolio) as w
                        WHERE {{starts_with('F', 'w::Symbol')}} = TRUE
                ) SELECT * FROM a WHERE LENGTH(a.Symbol) > 2
            )
        )
    }}
    """
    sql = """
    WITH a AS (
        SELECT * FROM (SELECT DISTINCT * FROM portfolio) as w
            WHERE w.Symbol LIKE 'F%'
    ) SELECT COUNT(*) FROM a WHERE LENGTH(a.Symbol) > 2
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["F"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_map_ingredient = db.execute_to_list(
        """
        SELECT COUNT(DISTINCT Symbol) FROM portfolio
        """
    )[0]
    passed_to_qa_ingredient = db.execute_to_list(
        """
    WITH a AS (
        SELECT * FROM (SELECT DISTINCT * FROM portfolio) as w
            WHERE w.Symbol LIKE 'F%'
    ) SELECT COUNT(*) FROM a WHERE LENGTH(a.Symbol) > 2
    """
    )[0]
    assert smoothie.meta.num_values_passed == (
        passed_to_map_ingredient + passed_to_qa_ingredient
    )


@pytest.mark.parametrize("db", databases)
def test_ingredient_in_select_with_join_multi_exec(db, ingredients):
    """If the query only has an ingredient in the `SELECT` statement, and `JOIN` clause,
    we should run the `JOIN` statement first, and then call the ingredient.

    commit de4a7bc
    """
    blendsql = """
    SELECT {{get_length('n_length', 'constituents::Name')}}
        FROM constituents JOIN account_history ON account_history.Symbol = constituents.Symbol
        WHERE account_history.Action like '%dividend%'
        ORDER BY constituents.Name
    """
    sql = """
    SELECT LENGTH(constituents.Name)
        FROM constituents JOIN account_history ON account_history.Symbol = constituents.Symbol
        WHERE account_history.Action like '%dividend%'
        ORDER BY constituents.Name
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT constituents.Name)
    FROM constituents JOIN account_history ON account_history.Symbol = constituents.Symbol
    WHERE account_history.Action like '%dividend%'
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_ingredient_in_select_with_join_multi_select_multi_exec(db, ingredients):
    """A modified version of the above

    commit de4a7bc
    """
    blendsql = """
    SELECT {{get_length('n_length', 'constituents::Name')}}, Action
        FROM constituents JOIN account_history ON account_history.Symbol = constituents.Symbol
        WHERE Action like '%dividend%'
    """
    sql = """
    SELECT LENGTH(constituents.Name), Action
        FROM constituents JOIN account_history ON account_history.Symbol = constituents.Symbol
        WHERE Action like '%dividend%'
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT constituents.Name)
    FROM constituents JOIN account_history ON account_history.Symbol = constituents.Symbol
    WHERE account_history.Action like '%dividend%'
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_subquery_alias_with_join_multi_exec(db, ingredients):
    blendsql = """
    SELECT w."Percent of Account" FROM (SELECT * FROM "portfolio" WHERE Quantity > 200 OR "Today's Gain/Loss Percent" > 0.05) as w
    JOIN {{
        do_join(
            left_on='geographic::Symbol',
            right_on='w::Symbol'
        )
    }} WHERE {{starts_with('F', 'w::Symbol')}}
    AND w."Percent of Account" < 0.2
    """

    sql = """
    SELECT w."Percent of Account" FROM (SELECT * FROM "portfolio" WHERE Quantity > 200 OR "Today's Gain/Loss Percent" > 0.05) as w
    JOIN geographic ON w.Symbol = geographic.Symbol
    WHERE w.Symbol LIKE 'F%'
    AND w."Percent of Account" < 0.2
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["F"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_to_list(
        """
    SELECT COUNT(DISTINCT Symbol) FROM portfolio WHERE (Quantity > 200 OR "Today's Gain/Loss Percent" > 0.05) AND "Percent of Account" < 0.2 
    """
    )[0]
    assert smoothie.meta.num_values_passed == passed_to_ingredient


@pytest.mark.parametrize("db", databases)
def test_materialize_ctes_multi_exec(db, ingredients):
    """We shouldn't create materialized CTE tables if they aren't used in an ingredient.

    commit dba7540
    """
    blendsql = """
    WITH a AS (
        SELECT * FROM portfolio WHERE Quantity > 200
    ), b AS (SELECT Symbol FROM portfolio AS w WHERE w.Symbol LIKE 'A%'),
    c AS (SELECT * FROM geographic)
    SELECT * FROM a WHERE {{starts_with('F', 'a::Symbol')}} = TRUE
    """
    # Need to get the lower level function so we can see what tables got created
    from blendsql.blend import _blend

    _ = _blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    assert db.has_temp_table("a")
    assert not db.has_temp_table("b")
    assert not db.has_temp_table("c")
    db._reset_connection()


@pytest.mark.parametrize("db", databases)
def test_options_referencing_cte_multi_exec(db, ingredients):
    """You should be able to reference a CTE in a QAIngredient `options` argument.

    f849ed3
    """
    blendsql = """
    WITH w AS (
        SELECT *
        FROM account_history
        WHERE Symbol IS NOT NULL
    ) SELECT {{select_first_sorted(options='w::Symbol')}} FROM w LIMIT 1
    """
    sql = """
        WITH w AS (
        SELECT *
        FROM account_history
        WHERE Symbol IS NOT NULL
    ) SELECT Symbol FROM w ORDER BY Symbol LIMIT 1
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


@pytest.mark.parametrize("db", databases)
def test_infer_options_arg(db, ingredients):
    """The infer_gen_constraints function should extend to cases when we do a
    `column = {{QAIngredient()}}` predicate.

    1a98559
    """
    blendsql = """
    SELECT * FROM account_history 
    WHERE Symbol = {{select_first_option()}}
    """
    sql = """
    SELECT * FROM account_history 
    WHERE Symbol = (SELECT Symbol FROM account_history WHERE Symbol NOT NULL ORDER BY Symbol LIMIT 1)
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_to_df(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)
