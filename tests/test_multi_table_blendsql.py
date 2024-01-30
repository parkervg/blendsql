import pytest
from tabulate import tabulate
from functools import partial
from blendsql import blend
from blendsql.db import SQLiteDBConnector
from tests.utils import (
    assert_equality,
    starts_with,
    get_length,
    select_first_sorted,
    do_join,
    return_aapl,
)

tabulate = partial(tabulate, headers="keys", showindex="never")


@pytest.fixture
def db() -> SQLiteDBConnector:
    return SQLiteDBConnector(db_path="./tests/data/multi_table.db")


@pytest.fixture
def ingredients() -> set:
    return {starts_with, get_length, select_first_sorted, do_join, return_aapl}


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
            WHERE portfolio.Description LIKE "A%"
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
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["A"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_query(
        """
    SELECT COUNT(DISTINCT Symbol) FROM portfolio WHERE Symbol in
    (
        SELECT Symbol FROM constituents
                WHERE sector = 'Information Technology'
    )
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


def test_join_multi_exec(db, ingredients):
    blendsql = """
    SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
        FROM account_history
        LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
        WHERE constituents.Sector = 'Information Technology'
        AND {{starts_with('A', 'constituents::Name')}} = 1
        AND lower(account_history.Action) like "%dividend%"
        """
    sql = """
    SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
        FROM account_history
        LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
        WHERE constituents.Sector = 'Information Technology'
        AND constituents.Name LIKE "A%"
        AND lower(account_history.Action) like "%dividend%"
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["A"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_query(
        """
    SELECT COUNT(DISTINCT Name) FROM constituents WHERE Sector = 'Information Technology'
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


def test_select_multi_exec(db, ingredients):
    blendsql = """
    SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name 
        FROM account_history 
        LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol 
        WHERE {{select_first_sorted(options='constituents::Sector')}}
        AND lower(account_history.Action) like "%dividend%"
    """
    sql = """
    SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
        FROM account_history
        LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
        WHERE constituents.Sector = (
            SELECT Sector FROM constituents 
            ORDER BY Sector LIMIT 1
        )
        AND lower(account_history.Action) like "%dividend%"
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


def test_complex_multi_exec(db, ingredients):
    blendsql = """
    SELECT DISTINCT constituents.Symbol as Symbol FROM constituents
    LEFT JOIN fundamentals ON constituents.Symbol = fundamentals.FPS_SYMBOL
    LEFT JOIN portfolio on constituents.Symbol = portfolio.Symbol
    WHERE fundamentals.MARKET_DAY_DT > '2023-02-23'
    AND ({{get_length('n_length', 'constituents::Name')}} > 3 OR {{starts_with('A', 'portfolio::Symbol')}})
    AND portfolio.Symbol IS NOT NULL
    ORDER BY {{get_length('n_length', 'constituents::Name')}} LIMIT 1
    """
    sql = """
    SELECT DISTINCT constituents.Symbol as Symbol FROM constituents
    LEFT JOIN fundamentals ON constituents.Symbol = fundamentals.FPS_SYMBOL
    LEFT JOIN portfolio on constituents.Symbol = portfolio.Symbol
    WHERE fundamentals.MARKET_DAY_DT > '2023-02-23'
    AND (LENGTH(constituents.Name) > 3 OR portfolio.Symbol LIKE "A%")
    AND portfolio.Symbol IS NOT NULL
    ORDER BY LENGTH(constituents.Name) LIMIT 1
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


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
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


def test_qa_equals_multi_exec(db, ingredients):
    blendsql = """
    SELECT Action FROM account_history
    WHERE Symbol = {{return_aapl()}} 
    """
    sql = """
    SELECT Action FROM account_history
    WHERE Symbol = "AAPL"
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df)


def test_table_alias_multi_exec(db, ingredients):
    blendsql = """
    SELECT Symbol FROM portfolio AS w
        WHERE {{starts_with('A', 'w::Symbol')}} = TRUE
        AND LENGTH(w.Symbol) > 3
    """
    sql = """
    SELECT Symbol FROM portfolio AS w
        WHERE w.Symbol LIKE "A%"
        AND LENGTH(w.Symbol) > 3
    """
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients=ingredients,
    )
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["A"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_query(
        """
    SELECT COUNT(DISTINCT Symbol) FROM portfolio WHERE LENGTH(Symbol) > 3
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


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
    sql_df = db.execute_query(sql)
    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["F"])
    # Make sure we only pass what's necessary to our ingredient
    passed_to_ingredient = db.execute_query(
        """
    SELECT COUNT(DISTINCT Symbol) FROM portfolio WHERE LENGTH(Symbol) > 3 AND Quantity > 200
    """
    )
    assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()


# def test_subquery_alias_with_join_multi_exec(db, ingredients):
#     blendsql = """
#     SELECT w."Percent of Account" FROM (SELECT * FROM "portfolio" WHERE Quantity > 200 OR "Today''s Gain/Loss Percent" > 0.05) as w
#     JOIN {{
#         do_join(
#             left_on='w::Symbol',
#             right_on='geographic::Symbol'
#         )
#     }} WHERE {{starts_with('F', 'w::Symbol')}}
#     AND w."Percent of Account" < 0.2
#     """
#
#     sql = """
#     SELECT w."Percent of Account" FROM (SELECT * FROM "portfolio" WHERE Quantity > 200 OR "Today''s Gain/Loss Percent" > 0.05) as w
#     JOIN geographic ON w.Symbol = geographic.Symbol
#     WHERE w.Symbol LIKE 'F%'
#     AND w."Percent of Account" < 0.2
#     """
#     smoothie = blend(
#         query=blendsql,
#         db=db,
#         ingredients=ingredients,
#     )
#     sql_df = db.execute_query(sql)
#     assert_equality(smoothie=smoothie, sql_df=sql_df, args=["F"])
#     # Make sure we only pass what's necessary to our ingredient
#     # passed_to_ingredient = db.execute_query(
#     #     """
#     # SELECT COUNT(DISTINCT Symbol) FROM portfolio WHERE LENGTH(Symbol) > 3
#     # """
#     # )
#     # assert smoothie.meta.num_values_passed == passed_to_ingredient.values[0].item()
