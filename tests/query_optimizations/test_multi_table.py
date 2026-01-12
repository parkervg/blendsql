import pytest

from blendsql import BlendSQL
from blendsql.common.utils import fetch_from_hub
from blendsql.db import DuckDB, SQLite
from tests.utils import (
    do_join,
    get_length,
    get_table_size,
    return_aapl,
    return_aapl_alias,
    select_first_option,
    select_first_sorted,
    test_starts_with,
)
from tests.query_optimizations.utils import TimedTestBase

dummy_ingredients = {
    test_starts_with,
    get_length,
    select_first_sorted,
    do_join,
    return_aapl,
    get_table_size,
    select_first_option,
    return_aapl_alias,
}


bsql_connections = [
    BlendSQL(SQLite(fetch_from_hub("multi_table.db")), ingredients=dummy_ingredients),
    BlendSQL(
        DuckDB.from_sqlite(fetch_from_hub("multi_table.db")),
        ingredients=dummy_ingredients,
    ),
]


class TestBasicOperations(TimedTestBase):
    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_simple_multi_exec(self, bsql: BlendSQL):
        """Test with multiple tables.
        Also ensures we only pass what is neccessary to the external ingredient F().
        "Show me the price of tech stocks in my portfolio that start with 'A'"
        """
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
            SELECT COUNT(DISTINCT Symbol) FROM portfolio WHERE Symbol in
            (
                SELECT Symbol FROM constituents
                WHERE sector = 'Information Technology'
            )
            """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT Symbol, "North America", "Japan" FROM geographic
                WHERE geographic.Symbol IN (
                    SELECT Symbol FROM portfolio
                    WHERE {{test_starts_with('A', Description)}} = 1
                    AND Symbol in (
                        SELECT Symbol FROM constituents
                        WHERE constituents.Sector = 'Information Technology'
                    )
                )
            """,
            sql_query="""
            SELECT Symbol, "North America", "Japan" FROM geographic
                WHERE geographic.Symbol IN (
                    SELECT Symbol FROM portfolio
                    WHERE portfolio.Description LIKE 'A%'
                    AND Symbol in (
                        SELECT Symbol FROM constituents
                        WHERE constituents.Sector = 'Information Technology'
                    )
                )
            """,
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_select_multi_exec(self, bsql: BlendSQL):
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
                FROM account_history
                JOIN constituents c ON account_history.Symbol = c.Symbol
                WHERE c.Sector = {{select_first_sorted(options=c.Sector)}}
                AND lower(LOWER(account_history.Action)) like '%dividend%'
                ORDER BY account_history."Run Date"
            """,
            sql_query="""
            SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
                FROM account_history
                JOIN constituents ON account_history.Symbol = constituents.Symbol
                WHERE constituents.Sector = (
                    SELECT Sector FROM constituents
                    ORDER BY Sector LIMIT 1
                )
                AND lower(LOWER(account_history.Action)) like '%dividend%'
                ORDER BY account_history."Run Date"
            """,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_complex_multi_exec(self, bsql: BlendSQL):
        """
        Below yields a tie in constituents.Name lengths, with 'Amgen' and 'Cisco'.
        DuckDB has different sorting behavior depending on the subset that's passed?
        """
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT DISTINCT Name FROM constituents
            JOIN account_history ON constituents.Symbol = account_history.Symbol
            JOIN portfolio on constituents.Symbol = portfolio.Symbol
            WHERE account_history."Run Date" > '2021-02-23'
            AND ({{get_length(constituents.Name)}} > 3 OR {{test_starts_with('A', portfolio.Symbol)}})
            AND portfolio.Symbol IS NOT NULL
            ORDER BY LENGTH(constituents.Name), constituents.Name LIMIT 4
            """,
            sql_query="""
            SELECT DISTINCT Name FROM constituents
            JOIN account_history ON constituents.Symbol = account_history.Symbol
            JOIN portfolio on constituents.Symbol = portfolio.Symbol
            WHERE account_history."Run Date" > '2021-02-23'
            AND (LENGTH(constituents.Name) > 3 OR portfolio.Symbol LIKE 'A%')
            AND portfolio.Symbol IS NOT NULL
            ORDER BY LENGTH(constituents.Name), constituents.Name LIMIT 4
            """,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_complex_not_qualified_multi_exec(self, bsql: BlendSQL):
        """Same test as test_complex_multi_exec(), but without qualifying columns if we don't need to.
        commit fefbc0a
        """
        passed_to_constituents_ingredient: int = bsql.db.execute_to_list(
            """
        SELECT COUNT(DISTINCT constituents.Name) FROM constituents
            LEFT JOIN account_history ON constituents.Symbol = account_history.Symbol
            LEFT JOIN portfolio on constituents.Symbol = portfolio.Symbol
            WHERE "Run Date" > '2021-02-23'
            AND portfolio.Symbol IS NOT NULL
        """,
            to_type=int,
        )[0]
        passed_to_portfolio_ingredient: int = bsql.db.execute_to_list(
            """
        SELECT COUNT(DISTINCT portfolio.Symbol) FROM constituents
            LEFT JOIN account_history ON constituents.Symbol = account_history.Symbol
            LEFT JOIN portfolio on constituents.Symbol = portfolio.Symbol
            WHERE "Run Date" > '2021-02-23'
            AND portfolio.Symbol IS NOT NULL
        """,
            to_type=int,
        )[0]
        expected_num_values_passed: int = (
            passed_to_portfolio_ingredient + passed_to_constituents_ingredient
        )

        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT DISTINCT constituents.Symbol, Action FROM constituents
            LEFT JOIN account_history ON constituents.Symbol = account_history.Symbol
            LEFT JOIN portfolio on constituents.Symbol = portfolio.Symbol
            WHERE "Run Date" > '2021-02-23'
            AND ({{get_length(constituents.Name)}} > 3 OR {{test_starts_with('A', portfolio.Symbol)}})
            AND portfolio.Symbol IS NOT NULL
            ORDER BY constituents.Symbol, LENGTH(Action) LIMIT 4
            """,
            sql_query="""
            SELECT DISTINCT constituents.Symbol, Action FROM constituents
            LEFT JOIN account_history ON constituents.Symbol = account_history.Symbol
            LEFT JOIN portfolio on constituents.Symbol = portfolio.Symbol
            WHERE "Run Date" > '2021-02-23'
            AND (LENGTH(constituents.Name) > 3 OR portfolio.Symbol LIKE 'A%')
            AND portfolio.Symbol IS NOT NULL
            ORDER BY constituents.Symbol, LENGTH(Action) LIMIT 4
            """,
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_qa_equals_multi_exec(self, bsql: BlendSQL):
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT Action FROM account_history
            WHERE Symbol = {{return_aapl()}}
            """,
            sql_query="""
            SELECT Action FROM account_history
            WHERE Symbol = 'AAPL'
            """,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_infer_options_arg(self, bsql: BlendSQL):
        """The infer_gen_constraints function should extend to cases when we do a
        `column = {{QAIngredient()}}` predicate.

        1a98559
        """
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT * FROM account_history
            WHERE Symbol = {{select_first_option()}}
            """,
            sql_query="""
            SELECT * FROM account_history
            WHERE Symbol = (SELECT Symbol FROM account_history WHERE Symbol NOT NULL ORDER BY Symbol LIMIT 1)
            """,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_null_negation(self, bsql: BlendSQL):
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT DISTINCT Name FROM constituents
            LEFT JOIN account_history ON constituents.Symbol = account_history.Symbol
            WHERE account_history."Settlement Date" IS NOT NULL
            AND {{test_starts_with('F', account_history.Account)}}
            ORDER BY constituents.Name
            """,
            sql_query="""
            SELECT DISTINCT Name FROM constituents
            LEFT JOIN account_history ON constituents.Symbol = account_history.Symbol
            WHERE account_history."Settlement Date" IS NOT NULL
            AND account_history.Account LIKE 'F%'
            ORDER BY constituents.Name
            """,
        )


class TestJoinOperations(TimedTestBase):
    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_join_multi_exec(self, bsql: BlendSQL):
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
        SELECT COUNT(DISTINCT Name) FROM account_history
        JOIN constituents ON account_history.Symbol = constituents.Symbol
        WHERE Sector = 'Information Technology'
        AND lower(Action) like '%dividend%'
        """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
                FROM account_history
                JOIN constituents c ON account_history.Symbol = c.Symbol
                WHERE c.Sector = 'Information Technology'
                AND {{test_starts_with('A', c.Name)}} = 1
                AND LOWER(account_history.Action) like '%dividend%'
                ORDER BY "Total Dividend Payout ($$)"
            """,
            sql_query="""
            SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
                FROM account_history
                JOIN constituents ON account_history.Symbol = constituents.Symbol
                WHERE constituents.Sector = 'Information Technology'
                AND constituents.Name LIKE 'A%'
                AND LOWER(account_history.Action) like '%dividend%'
                ORDER BY "Total Dividend Payout ($$)"
            """,
            args=["A"],
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_join_not_qualified_multi_exec(self, bsql: BlendSQL):
        """Same test as test_join_multi_exec(), but without qualifying columns if we don't need to.
        i.e. 'Action' and 'Sector' don't have tablename preceding them.
        commit fefbc0a
        """
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
        SELECT COUNT(DISTINCT Name) FROM account_history
        JOIN constituents ON account_history.Symbol = constituents.Symbol
        WHERE Sector = 'Information Technology'
        AND lower(Action) like '%dividend%'
        """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
                FROM account_history
                JOIN constituents ON account_history.Symbol = constituents.Symbol
                WHERE Sector = 'Information Technology'
                AND {{test_starts_with('A', constituents.Name)}} = 1
                AND lower(Action) like '%dividend%'
                ORDER BY "Run Date"
            """,
            sql_query="""
            SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
                FROM account_history
                JOIN constituents ON account_history.Symbol = constituents.Symbol
                WHERE Sector = 'Information Technology'
                AND constituents.Name LIKE 'A%'
                AND lower(Action) like '%dividend%'
                ORDER BY "Run Date"
            """,
            args=["A"],
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_join_ingredient_multi_exec(self, bsql: BlendSQL):
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT Account, Quantity FROM returns r
            JOIN account_history a ON {{
                do_join(
                    r."Annualized Returns",
                    a.Account
                )
            }}
            """,
            sql_query="""
            SELECT Account, Quantity FROM returns JOIN account_history ON returns."Annualized Returns" = account_history."Account"
            """,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_join_with_multiple_ingredients(self, bsql: BlendSQL):
        """
        af86714
        """
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT "Run Date", Action, p.Symbol FROM account_history a
            JOIN portfolio p ON {{
                do_join(
                    a.Symbol,
                    p.Symbol
                )
            }} WHERE {{
                test_starts_with('H', p.Description)
            }} AND {{
                get_length(a."Security Description")
            }} > 3""",
            sql_query="""
            SELECT "Run Date", Action, portfolio.Symbol FROM account_history
            JOIN portfolio ON account_history.Symbol = portfolio.Symbol
            WHERE portfolio.Description LIKE 'H%'
            AND LENGTH(account_history."Security Description") > 3
            """,
        )


class TestAliasOperations(TimedTestBase):
    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_table_alias_multi_exec(self, bsql: BlendSQL):
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
            SELECT COUNT(DISTINCT Symbol) FROM portfolio WHERE LENGTH(Symbol) > 3
            """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT Symbol FROM portfolio AS w
                WHERE {{test_starts_with('A', w.Symbol)}}
                AND LENGTH(w.Symbol) > 3
            """,
            sql_query="""
            SELECT Symbol FROM portfolio AS w
                WHERE w.Symbol LIKE 'A%'
                AND LENGTH(w.Symbol) > 3
            """,
            expected_num_values_passed=expected_num_values_passed,
            args=["A"],
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_subquery_alias_multi_exec(self, bsql: BlendSQL):
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
            SELECT COUNT(DISTINCT Symbol) FROM portfolio WHERE LENGTH(Symbol) > 3 AND Quantity > 200
            """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT Symbol FROM (
                SELECT DISTINCT Symbol FROM portfolio WHERE Symbol IN (
                    SELECT Symbol FROM portfolio WHERE Quantity > 200
                )
            ) AS w WHERE {{test_starts_with('F', w.Symbol)}} AND LENGTH(w.Symbol) > 3
            """,
            sql_query="""
            SELECT Symbol FROM (
                SELECT DISTINCT Symbol FROM portfolio WHERE Symbol IN (
                    SELECT Symbol FROM portfolio WHERE Quantity > 200
                )
            ) AS w WHERE w.Symbol LIKE 'F%' AND LENGTH(w.Symbol) > 3
            """,
            expected_num_values_passed=expected_num_values_passed,
        )


class TestCTEOperations(TimedTestBase):
    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_cte_qa_multi_exec(self, bsql: BlendSQL):
        passed_to_map_ingredient: int = bsql.db.execute_to_list(
            """
        SELECT COUNT(DISTINCT Symbol) FROM portfolio
        """,
            to_type=int,
        )[0]
        # We also need to factor in what we passed to QA ingredient
        passed_to_qa_ingredient: int = bsql.db.execute_to_list(
            """
        WITH a AS (
            SELECT * FROM (SELECT DISTINCT * FROM portfolio) as w
                WHERE w.Symbol LIKE 'F%'
        ) SELECT COUNT(*) FROM a WHERE LENGTH(a.Symbol) > 2
        """,
            to_type=int,
        )[0]
        expected_num_values_passed: int = (
            passed_to_qa_ingredient + passed_to_map_ingredient
        )

        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            {{
                 get_table_size(
                     context=(
                         WITH a AS (
                             SELECT * FROM (SELECT DISTINCT * FROM portfolio) as w
                                 WHERE {{test_starts_with('F', w.Symbol)}} = TRUE
                         ) SELECT * FROM a WHERE LENGTH(a.Symbol) > 2
                     )
                 )
             }}
            """,
            sql_query="""
            WITH a AS (
                SELECT * FROM (SELECT DISTINCT * FROM portfolio) as w
                    WHERE w.Symbol LIKE 'F%'
            ) SELECT COUNT(*) FROM a WHERE LENGTH(a.Symbol) > 2
            """,
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_cte_qa_named_multi_exec(self, bsql: BlendSQL):
        passed_to_map_ingredient: int = bsql.db.execute_to_list(
            """
            SELECT COUNT(DISTINCT Symbol) FROM portfolio
            """,
            to_type=int,
        )[0]
        passed_to_qa_ingredient: int = bsql.db.execute_to_list(
            """
        WITH a AS (
            SELECT * FROM (SELECT DISTINCT * FROM portfolio) as w
                WHERE w.Symbol LIKE 'F%'
        ) SELECT COUNT(*) FROM a WHERE LENGTH(a.Symbol) > 2
        """,
            to_type=int,
        )[0]
        expected_num_values_passed: int = (
            passed_to_map_ingredient + passed_to_qa_ingredient
        )
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            {{
                 get_table_size(
                     context=(
                         WITH a AS (
                             SELECT * FROM (SELECT DISTINCT * FROM portfolio) as w
                                 WHERE {{test_starts_with('F', w.Symbol)}} = TRUE
                         ) SELECT * FROM a WHERE LENGTH(a.Symbol) > 2
                     )
                 )
             }}
            """,
            sql_query="""
            WITH a AS (
                SELECT * FROM (SELECT DISTINCT * FROM portfolio) as w
                    WHERE w.Symbol LIKE 'F%'
            ) SELECT COUNT(*) FROM a WHERE LENGTH(a.Symbol) > 2
            """,
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_materialize_ctes_multi_exec(self, bsql: BlendSQL):
        """We shouldn't create materialized CTE tables if they aren't used in an ingredient.

        commit dba7540
        """
        blendsql = """
        WITH a AS (
            SELECT * FROM portfolio WHERE Quantity > 200
        ), b AS (SELECT Symbol FROM portfolio AS w WHERE w.Symbol LIKE 'A%'),
        c AS (SELECT * FROM geographic)
        SELECT * FROM a WHERE {{test_starts_with('F', Symbol)}}
        """
        # Need to get the lower level function so we can see what tables got created
        from blendsql.blendsql import _blend

        _ = _blend(
            query=blendsql,
            db=bsql.db,
            ingredients=dummy_ingredients,
        )
        assert bsql.db.has_temp_table("a")
        assert not bsql.db.has_temp_table("b")
        assert not bsql.db.has_temp_table("c")
        bsql.db._reset_connection()

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_options_referencing_cte_multi_exec(self, bsql: BlendSQL):
        """You should be able to reference a CTE in a QAIngredient `options` argument.

        f849ed3
        """
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            WITH w AS (
                SELECT *
                FROM account_history
                WHERE Symbol IS NOT NULL
            ) SELECT {{select_first_sorted(options=Symbol)}} FROM w LIMIT 1
            """,
            sql_query="""
            WITH w AS (
                SELECT *
                FROM account_history
                WHERE Symbol IS NOT NULL
            ) SELECT Symbol FROM w ORDER BY Symbol LIMIT 1
            """,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_not_double_executing_cte(self, bsql: BlendSQL):
        passed_to_ingredient_1 = bsql.db.execute_to_list(
            """
            SELECT COUNT(DISTINCT Account)
            FROM account_history
            WHERE Account IS NOT NULL
            """,
            to_type=int,
        )[0]
        passed_to_ingredient_2: int = bsql.db.execute_to_list(
            """
            SELECT COUNT(DISTINCT Symbol)
            FROM account_history
            WHERE Account LIKE 'F%'
            AND Symbol IS NOT NULL
            """,
            to_type=int,
        )[0]
        expected_num_values_passed: int = (
            passed_to_ingredient_1 + passed_to_ingredient_2
        )
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            WITH t AS (
                SELECT Symbol FROM account_history a
                WHERE {{test_starts_with('F', a.Account)}} = TRUE
                AND a.Account IS NOT NULL
            ) SELECT COUNT(*) FROM t
            WHERE Symbol IS NOT NULL
            AND {{get_length(Symbol)}} > 3
            """,
            sql_query="""
            WITH t AS (
                SELECT Symbol FROM account_history a
                WHERE Account LIKE 'F%'
                AND Account IS NOT NULL
            ) SELECT COUNT(*) FROM t
            WHERE t.Symbol IS NOT NULL
            AND LENGTH(t.Symbol) > 3
            """,
            expected_num_values_passed=expected_num_values_passed,
        )


class TestSelectOperations(TimedTestBase):
    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_ingredient_in_select_with_join_multi_exec(self, bsql: BlendSQL):
        """If the query only has an ingredient in the `SELECT` statement, and `JOIN` clause,
        we should run the `JOIN` statement first, and then call the ingredient.

        commit de4a7bc
        """
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
            SELECT COUNT(DISTINCT constituents.Name)
            FROM constituents JOIN account_history ON account_history.Symbol = constituents.Symbol
            WHERE LOWER(account_history.Action) like '%dividend%'
            """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT {{get_length(c.Name)}}
                FROM constituents c JOIN account_history a ON a.Symbol = c.Symbol
                WHERE LOWER(a.Action) like '%dividend%'
                ORDER BY c.Name
            """,
            sql_query="""
            SELECT LENGTH(constituents.Name)
                FROM constituents JOIN account_history ON account_history.Symbol = constituents.Symbol
                WHERE LOWER(account_history.Action) like '%dividend%'
                ORDER BY constituents.Name
            """,
            expected_num_values_passed=expected_num_values_passed,
        )
        smoothie = bsql.execute(
            """
            SELECT {{get_length(c.Name)}}
                FROM constituents c JOIN account_history a ON a.Symbol = c.Symbol
                WHERE LOWER(a.Action) like '%dividend%'
                ORDER BY c.Name
            """
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_ingredient_in_select_with_join_multi_select_multi_exec(
        self, bsql: BlendSQL
    ):
        """A modified version of the above

        commit de4a7bc
        """
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
            SELECT COUNT(DISTINCT constituents.Name)
            FROM constituents JOIN account_history ON account_history.Symbol = constituents.Symbol
            WHERE LOWER(account_history.Action) like '%dividend%'
            """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT {{get_length(constituents.Name)}}, Action
                FROM constituents JOIN account_history ON account_history.Symbol = constituents.Symbol
                WHERE LOWER(Action) like '%dividend%'
            """,
            sql_query="""
            SELECT LENGTH(constituents.Name), Action
                FROM constituents JOIN account_history ON account_history.Symbol = constituents.Symbol
                WHERE LOWER(Action) like '%dividend%'
            """,
            expected_num_values_passed=expected_num_values_passed,
        )


class TestSubqueryOperations(TimedTestBase):
    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_subquery_alias_with_join_multi_exec(self, bsql: BlendSQL):
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
            SELECT COUNT(DISTINCT Symbol) FROM portfolio WHERE (Quantity > 200 OR "Today's Gain/Loss Percent" > 0.05) AND "Percent of Account" < 0.2
            """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT w."Percent of Account" FROM (SELECT * FROM "portfolio" WHERE Quantity > 200 OR "Today's Gain/Loss Percent" > 0.05) as w
            JOIN geographic g ON {{
                do_join(
                    w.Symbol,
                    g.Symbol
                )
            }} WHERE {{test_starts_with('F', w.Symbol)}}
            AND w."Percent of Account" < 0.2
            """,
            sql_query="""
            SELECT w."Percent of Account" FROM (SELECT * FROM "portfolio" WHERE Quantity > 200 OR "Today's Gain/Loss Percent" > 0.05) as w
            JOIN geographic ON w.Symbol = geographic.Symbol
            WHERE w.Symbol LIKE 'F%'
            AND w."Percent of Account" < 0.2
            """,
            expected_num_values_passed=expected_num_values_passed,
            args=["F"],
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_subquery_alias_with_join_multi_exec_and(self, bsql: BlendSQL):
        """Same as before, but now we use an `AND` predicate to link the `JOIN` and `LIKE` clauses.
        This will hit the `replace_join_with_ingredient_multiple_ingredient` transform.
        """
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
            SELECT COUNT(DISTINCT Symbol) FROM portfolio WHERE (Quantity > 200 OR "Today's Gain/Loss Percent" > 0.05)
            """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT w."Percent of Account" FROM (SELECT * FROM "portfolio" WHERE Quantity > 200 OR "Today's Gain/Loss Percent" > 0.05) as w
            JOIN geographic g ON {{
                do_join(
                    w.Symbol,
                    g.Symbol
                )
            }} WHERE {{test_starts_with('F', w.Symbol)}}
            """,
            sql_query="""
            SELECT w."Percent of Account" FROM (SELECT * FROM "portfolio" WHERE Quantity > 200 OR "Today's Gain/Loss Percent" > 0.05) as w
            JOIN geographic ON w.Symbol = geographic.Symbol
            AND w.Symbol LIKE 'F%'
            """,
            expected_num_values_passed=expected_num_values_passed,
            args=["F"],
        )


# def test_alias_ingredient_multi_exec(bsql):
#     """Tests the AliasIngredient class.
#     We don't inherit the dummy_ingredients fixture here,
#     since we want to be sure the 'return_aapl_alias' ingredient
#     is correctly injecting any dependent ingredients at runtime.
#
#     commit 6bac71f
#     """
#     smoothie = bsql.execute(
#         """
#         SELECT Action FROM account_history
#         WHERE Symbol = {{return_aapl_alias()}}
#         """,
#         ingredients={return_aapl_alias},
#     )
#     sql_df = bsql.db.execute_to_df(
#         """
#         SELECT Action FROM account_history
#         WHERE Symbol = 'AAPL'
#         """
#     )
#     assert_equality(smoothie=smoothie, sql_df=sql_df)


#
# def test_alias_tuple_ingredient_multi_exec(bsql):
#     """
#     commit d795a00
#     """
#     smoothie = bsql.execute(
#         """
#         SELECT Symbol FROM portfolio AS w
#             WHERE {{test_starts_with('A', w.Symbol)}} = TRUE
#             AND Symbol IN {{return_stocks_tuple_alias()}}
#             AND LENGTH(w.Symbol) > 3
#         """,
#         ingredients={test_starts_with, return_stocks_tuple_alias},
#     )
#     sql_df = bsql.db.execute_to_df(
#         """
#         SELECT Symbol FROM portfolio AS w
#             WHERE w.Symbol LIKE 'A%'
#             AND Symbol IN ('AAPL', 'AMZN', 'TYL')
#             AND LENGTH(w.Symbol) > 3
#         """
#     )
#     assert_equality(smoothie=smoothie, sql_df=sql_df)
#     # Make sure we only pass what's necessary to our ingredient
#     passed_to_ingredient = bsql.db.execute_to_list(
#         """
#     SELECT COUNT(DISTINCT Symbol) FROM portfolio WHERE LENGTH(Symbol) > 3
#     """
#     )[0]
#     assert smoothie.meta.num_values_passed == passed_to_ingredient
