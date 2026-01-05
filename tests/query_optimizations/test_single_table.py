import pytest

from blendsql import BlendSQL
from blendsql.common.utils import fetch_from_hub
from blendsql.db import DuckDB, SQLite
from tests.utils import (
    get_length,
    get_table_size,
    select_first_option,
    select_first_sorted,
    test_starts_with,
)
from tests.query_optimizations.utils import TimedTestBase


dummy_ingredients = {
    test_starts_with,
    get_length,
    select_first_sorted,
    get_table_size,
    select_first_option,
}


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


class TestBasicOperations(TimedTestBase):
    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_simple_exec(self, bsql: BlendSQL):
        smoothie = bsql.execute(
            """
            SELECT * FROM transactions;
            """
        )
        assert not smoothie.df.empty

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_nested_exec(self, bsql: BlendSQL):
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
    def test_simple_ingredient_exec(self, bsql: BlendSQL):
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT * FROM transactions WHERE {{test_starts_with('Z', merchant)}} = 1;
            """,
            sql_query="""
            SELECT * FROM transactions WHERE merchant LIKE 'Z%';
            """,
            args=["Z"],
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_simple_ingredient_exec_at_end(self, bsql: BlendSQL):
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT * FROM transactions WHERE {{test_starts_with('Z', merchant)}}
            """,
            sql_query="""
            SELECT * FROM transactions WHERE merchant LIKE 'Z%';
            """,
            args=["Z"],
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_nested_ingredient_exec(self, bsql: BlendSQL):
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
                SELECT COUNT(DISTINCT merchant) FROM transactions WHERE amount > 100
                """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT DISTINCT merchant FROM transactions WHERE
                merchant in (
                    SELECT merchant FROM transactions
                        WHERE amount > 100
                        AND {{test_starts_with('Z', merchant)}} = 1
                );
            """,
            sql_query="""
            SELECT DISTINCT merchant FROM transactions WHERE
               merchant in (
                   SELECT merchant FROM transactions
                       WHERE amount > 100
                       AND merchant LIKE 'Z%'
               );
            """,
            expected_num_values_passed=expected_num_values_passed,
            args=["Z"],
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_nonexistent_column_exec(self, bsql: BlendSQL):
        """
        NOTE: Converting to CNF would break this one
        since we would get:
            SELECT DISTINCT merchant, child_category FROM transactions WHERE
            (child_category = 'Gifts' OR STRUCT(STRUCT(A())) = 1) AND
            (child_category = 'Gifts' OR child_category = 'this does not exist')
        """
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
            SELECT COUNT(DISTINCT merchant) FROM transactions WHERE child_category = 'this does not exist'

            """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT DISTINCT merchant, child_category FROM transactions WHERE
               (
                   {{test_starts_with('Z', merchant)}} = 1
                   AND child_category = 'this does not exist'
               )
               OR child_category = 'Gifts'
               ORDER BY merchant
            """,
            sql_query="""
            SELECT DISTINCT merchant, child_category FROM transactions WHERE
               (
                   merchant LIKE 'Z%'
                   AND child_category = 'this does not exist'
               )
               OR child_category = 'Gifts'
               ORDER BY merchant
            """,
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_nested_and_exec(self, bsql: BlendSQL):
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
                SELECT COUNT(DISTINCT merchant) FROM transactions WHERE child_category = 'Restaurants & Dining'
                """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT DISTINCT merchant, child_category FROM transactions WHERE
               (
                   {{test_starts_with('O', merchant)}} = 1
                   AND child_category = 'Restaurants & Dining'
               )
               OR child_category = 'Gifts'
               ORDER BY merchant
            """,
            sql_query="""
            SELECT DISTINCT merchant, child_category FROM transactions WHERE
               (
                   merchant LIKE 'O%'
                   AND child_category = 'Restaurants & Dining'
               )
               OR child_category = 'Gifts'
               ORDER BY merchant
            """,
            expected_num_values_passed=expected_num_values_passed,
            args=["O"],
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_multiple_nested_ingredients(self, bsql: BlendSQL):
        """This one benefits from cascade filtering"""
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
                SELECT (
                    SELECT COUNT(DISTINCT merchant) FROM transactions
                    WHERE parent_category = 'Food'
                ) + (
                    SELECT COUNT(DISTINCT child_category) FROM transactions
                    WHERE parent_category = 'Food' 
                    AND merchant LIKE 'A%'
                )
                """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT DISTINCT child_category, merchant FROM transactions WHERE
                (
                    {{test_starts_with('A', merchant)}} = 1
                    AND {{test_starts_with('T', child_category)}} = 1
                    AND parent_category = 'Food'
                )
               OR child_category = 'Gifts'
               ORDER BY merchant
            """,
            sql_query="""
            SELECT DISTINCT child_category, merchant FROM transactions WHERE
                (
                    merchant LIKE 'A%'
                    AND child_category LIKE 'T%'
                    AND parent_category = 'Food'
                )
                OR child_category = 'Gifts'
                ORDER BY merchant
            """,
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_length_ingredient(self, bsql: BlendSQL):
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
            SELECT COUNT(DISTINCT merchant) FROM transactions
            """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT {{get_length(merchant)}}, merchant
                FROM transactions
                WHERE {{get_length(merchant)}} > 1;
            """,
            sql_query="""
            SELECT LENGTH(merchant) as length, merchant
                FROM transactions
                WHERE LENGTH(merchant) > 1;
            """,
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_max_length(self, bsql: BlendSQL):
        """In DuckDB, this causes
        `Binder Error: aggregate function calls cannot be nested`
        """
        if isinstance(bsql.db, DuckDB):
            pytest.skip("Skipping nested aggregate for DuckDB...")

        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
                SELECT COUNT(DISTINCT merchant) FROM transactions
                """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT MAX({{get_length(merchant)}}) as max_length, merchant
                FROM transactions
                WHERE {{get_length(merchant)}} > 1;
            """,
            sql_query="""
            SELECT MAX(LENGTH(merchant)) as max_length, merchant
                FROM transactions
                WHERE LENGTH(merchant) > 1;
            """,
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_nested_duplicate_map_calls(self, bsql: BlendSQL):
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
                SELECT COUNT(DISTINCT merchant) FROM transactions
                """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT merchant FROM transactions WHERE {{get_length(merchant)}} > (SELECT {{get_length(merchant)}} FROM transactions WHERE merchant = 'Paypal' LIMIT 1)
            """,
            sql_query="""
            SELECT merchant FROM transactions WHERE LENGTH(merchant) > (SELECT LENGTH(merchant) FROM transactions WHERE merchant = 'Paypal' LIMIT 1)
            """,
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_many_duplicate_map_calls(self, bsql: BlendSQL):
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
                SELECT
                (SELECT COUNT(DISTINCT merchant) FROM transactions WHERE amount > 1300)
                + (SELECT COUNT(DISTINCT cash_flow) FROM transactions WHERE amount > 1300)
                + (SELECT COUNT(DISTINCT child_category) FROM transactions WHERE amount > 1300)
                + (SELECT COUNT(DISTINCT date) FROM transactions WHERE amount > 1300)
                """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT
                {{get_length(merchant)}} AS l1,
                {{get_length(cash_flow)}} AS l2,
                {{get_length(child_category)}} AS l3,
                {{get_length(date)}} AS l4
            FROM transactions WHERE amount > 1300
            """,
            sql_query="""
            SELECT
                LENGTH(merchant) AS l1,
                LENGTH(cash_flow) AS l2,
                LENGTH(child_category) AS l3,
                LENGTH(date) AS l4
            FROM transactions WHERE amount > 1300
            """,
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_exists_isolated_qa_call(self, bsql: BlendSQL):
        # commit 7a19e39
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
                SELECT (SELECT COUNT(DISTINCT merchant) FROM transactions WHERE amount > 500) + (SELECT COUNT(*) FROM transactions WHERE amount < 500)
                """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT NOT EXISTS (
                SELECT * FROM transactions WHERE {{get_length( merchant)}} > 4 AND amount > 500
            ) OR (
                {{
                    get_table_size((select * from transactions where amount < 500))
                }}
            )
            """,
            sql_query="""
            SELECT NOT EXISTS (
                SELECT * FROM transactions WHERE
                LENGTH(merchant) > 4 AND amount > 500
            ) OR (
                select count(*) from transactions where amount < 500
            )
            """,
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_query_options_arg(self, bsql: BlendSQL):
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
    def test_filter_by_ingredient(self, bsql: BlendSQL):
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT DISTINCT merchant, child_category FROM transactions WHERE
               merchant = {{select_first_sorted(options=merchant)}}
            """,
            sql_query="""
            SELECT DISTINCT merchant, child_category FROM transactions
                ORDER BY merchant LIMIT 1
            """,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_where_with_true(self, bsql: BlendSQL):
        """Makes sure we don't ignore `{column} = TRUE` SQL clauses."""
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
            SELECT COUNT(DISTINCT merchant) FROM transactions WHERE LENGTH(merchant) = 5
            """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            WITH a AS (
                SELECT LENGTH(merchant) = 5 AS b, merchant FROM transactions
            )
            SELECT merchant FROM a WHERE a.b = TRUE AND {{test_starts_with('Z', merchant)}}
            """,
            sql_query="""
            WITH a AS (
                SELECT LENGTH(merchant) = 5 AS b, merchant FROM transactions
            )
            SELECT merchant FROM a WHERE a.b = TRUE AND a.merchant LIKE 'Z%'
            """,
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_where_in_clause(self, bsql: BlendSQL):
        """Makes sure we don't ignore `{column} = TRUE` SQL clauses."""
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT merchant FROM transactions WHERE merchant IN (SELECT merchant FROM transactions WHERE {{test_starts_with('Z', merchant)}})
            """,
            sql_query="""
            SELECT merchant FROM transactions WHERE merchant IN (SELECT merchant FROM transactions WHERE merchant LIKE 'Z%')
            """,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_null_negation(self, bsql: BlendSQL):
        """ee3b0c4"""
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT merchant FROM transactions
            WHERE merchant IS NOT NULL
            AND {{test_starts_with('Z', merchant)}}
            """,
            sql_query="""
            SELECT merchant FROM transactions
            WHERE merchant IS NOT NULL
            AND merchant LIKE 'Z%'
            """,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_map_in_function(self, bsql: BlendSQL):
        """6cec1a4"""
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
                SELECT COUNT(DISTINCT merchant) FROM transactions WHERE merchant LIKE 'Z%'
                """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT merchant FROM transactions t
            WHERE LENGTH(CAST({{get_length(merchant)}} AS TEXT)) = 1
            AND merchant LIKE 'Z%'
            """,
            sql_query="""
            SELECT merchant FROM transactions t
            WHERE LENGTH(CAST(LENGTH(t.merchant) AS TEXT)) = 1
            AND merchant LIKE 'Z%'
            """,
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_concat_in_map(self, bsql: BlendSQL):
        """dcc87bb

        After adding cascade filtering, this query only passes 137 values to ingredients,
            as opposed to 274.
        """
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
                SELECT (
                    SELECT COUNT(DISTINCT merchant || ' ' || child_category) FROM transactions 
                    WHERE child_category LIKE 'P%'
                ) + (
                    SELECT COUNT(DISTINCT merchant) FROM transactions 
                    WHERE LENGTH(merchant || ' ' || child_category) > 50
                    AND child_category LIKE 'P%'
                )
                """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
                SELECT merchant FROM transactions t
                WHERE {{get_length(t.merchant || ' ' || t.child_category)}} > 50
                AND {{test_starts_with('C', t.merchant)}}
                AND t.child_category LIKE 'P%'
                """,
            sql_query="""
                SELECT DISTINCT merchant FROM transactions t 
                WHERE LENGTH(t.merchant || ' ' || t.child_category) > 50 
                AND t.merchant LIKE 'C%' 
                AND t.child_category LIKE 'P%'
                """,
            expected_num_values_passed=expected_num_values_passed,
        )


class TestSelectOperations(TimedTestBase):
    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_ingredient_in_select_stmt(self, bsql: BlendSQL):
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
                SELECT COUNT(DISTINCT merchant) FROM transactions
                """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT MAX({{get_length(merchant)}}) as l FROM transactions
            """,
            sql_query="""
            SELECT MAX(LENGTH(merchant)) as l FROM transactions
            """,
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_simple_ingredient_exec_in_select(self, bsql: BlendSQL):
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
                SELECT COUNT(DISTINCT merchant) FROM transactions WHERE parent_category = 'Auto & Transport'
                """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT {{get_length(merchant)}} AS "LENGTH(merchant)" FROM transactions WHERE parent_category = 'Auto & Transport'
            """,
            sql_query="""
            SELECT LENGTH(merchant) FROM transactions WHERE parent_category = 'Auto & Transport'
            """,
            expected_num_values_passed=expected_num_values_passed,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_ingredient_in_select_stmt_with_filter(self, bsql: BlendSQL):
        # commit de4a7bc
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
                SELECT COUNT(DISTINCT merchant) FROM transactions WHERE child_category = 'Restaurants & Dining'
                """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT MAX({{get_length(merchant)}}) as l FROM transactions WHERE child_category = 'Restaurants & Dining'
            """,
            sql_query="""
            SELECT MAX(LENGTH(merchant)) as l FROM transactions WHERE child_category = 'Restaurants & Dining'
            """,
            expected_num_values_passed=expected_num_values_passed,
        )


class TestLimitOperations(TimedTestBase):
    # @pytest.mark.parametrize("bsql", bsql_connections)
    # def test_limit(self, bsql: BlendSQL):
    #     expected_num_values_passed: int = bsql.db.execute_to_list(
    #         """
    #             SELECT COUNT(DISTINCT merchant) FROM transactions WHERE child_category = 'Restaurants & Dining'
    #             """,
    #         to_type=int,
    #     )[0]
    #     _ = self.assert_blendsql_equals_sql(
    #         bsql,
    #         blendsql_query="""
    #         SELECT DISTINCT merchant, child_category FROM transactions WHERE
    #         {{test_starts_with('P', merchant)}} = 1
    #         AND child_category = 'Restaurants & Dining'
    #         ORDER BY merchant
    #         LIMIT 1
    #         """,
    #         sql_query="""
    #         SELECT DISTINCT merchant, child_category FROM transactions WHERE
    #             merchant LIKE 'P%'
    #             AND child_category = 'Restaurants & Dining'
    #             ORDER BY merchant
    #             LIMIT 1
    #         """,
    #         expected_num_values_passed=expected_num_values_passed,
    #     )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_ingredient_in_select_with_limit(self, bsql: BlendSQL):
        # commit 335c67a
        smoothie = bsql.execute(
            """
            SELECT {{get_length(merchant)}} FROM transactions ORDER BY merchant LIMIT 1
            """
        )
        assert smoothie.meta.num_values_passed == 1

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_apply_limit_with_predicate(self, bsql: BlendSQL):
        # commit 335c67a
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
                SELECT COUNT(DISTINCT merchant) FROM transactions WHERE amount > 1300 LIMIT 3
                """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT {{get_length(merchant)}}
            FROM transactions
            WHERE amount > 1300
            ORDER BY merchant LIMIT 3
            """,
            sql_query="""
            SELECT LENGTH(merchant)
            FROM transactions
            WHERE amount > 1300
            ORDER BY merchant LIMIT 3
            """,
            expected_num_values_passed=expected_num_values_passed,
            # We say `<=` here because the ingredient operates over sets, rather than lists
            # So this kind of screws up the `LIMIT` calculation
            # But execution outputs should be the same (tested above)
            allow_lt_num_values_compare=True,
        )


class TestGroupByOperations(TimedTestBase):
    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_group_by_with_ingredient_alias(self, bsql: BlendSQL):
        """b28a129"""
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT SUM(amount) AS "total amount",
            {{get_length(merchant)}} AS "Merchant Length"
            FROM transactions
            GROUP BY "Merchant Length"
            ORDER BY "total amount"
            """,
            sql_query="""
            SELECT SUM(amount) AS "total amount",
            LENGTH(merchant) AS "Merchant Length"
            FROM transactions
            GROUP BY "Merchant Length"
            ORDER BY "total amount"
            """,
        )


class TestCTEOperations(TimedTestBase):
    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_cte_with_ingredient(self, bsql: BlendSQL):
        """c3ec1eb"""
        _ = bsql.execute(
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

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_multiple_ctes_with_ingredient(self, bsql: BlendSQL):
        # First just make sure a `SELECT *` query works
        _ = bsql.execute(
            """
            WITH a AS (
                SELECT merchant, {{get_length(merchant)}} FROM transactions
            ), b AS (
                SELECT merchant FROM transactions
            ) SELECT * FROM a JOIN b ON a.merchant = b.merchant
            """
        )
        # Then make sure results are correct
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            WITH a AS (
                SELECT merchant, {{get_length(merchant)}} AS length_output FROM transactions
            ), b AS (
                SELECT merchant FROM transactions
            ) SELECT a.merchant, length_output FROM a JOIN b ON a.merchant = b.merchant
            """,
            sql_query="""
            WITH a AS (
                SELECT merchant, LENGTH(merchant) AS length_output FROM transactions
            ), b AS (
                SELECT merchant FROM transactions
            ) SELECT a.merchant, length_output FROM a JOIN b ON a.merchant = b.merchant
            """,
        )


class TestOffsetOperations(TimedTestBase):
    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_offset(self, bsql: BlendSQL):
        """SELECT "fruits"."name" FROM fruits OFFSET 2 will break SQLite
        for some reason, DuckDB is ok with it, though.
        """
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT merchant FROM transactions t
            WHERE {{test_starts_with('Z', merchant)}}
            ORDER BY {{get_length(t.merchant)}} LIMIT 1 OFFSET 2
            """,
            sql_query="""
            SELECT merchant FROM transactions t
            WHERE merchant LIKE 'Z%'
            ORDER BY LENGTH(merchant) LIMIT 1 OFFSET 2
            """,
        )


class TestHavingOperations(TimedTestBase):
    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_having(self, bsql: BlendSQL):
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
                SELECT COUNT(DISTINCT merchant) FROM transactions WHERE merchant LIKE 'Z%'
                """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT {{get_length(merchant)}} AS l FROM transactions t
            WHERE merchant LIKE 'Z%'
            GROUP BY merchant, l
            HAVING l > 3
            ORDER BY l
            """,
            sql_query="""
            SELECT LENGTH(merchant) AS l FROM transactions t
            WHERE merchant LIKE 'Z%'
            GROUP BY merchant
            HAVING l > 3
            ORDER BY l
            """,
            expected_num_values_passed=expected_num_values_passed,
        )


if __name__ == "__main__":
    pytest.main()
