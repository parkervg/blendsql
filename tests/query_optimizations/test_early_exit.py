import pytest
import pandas as pd
from blendsql import BlendSQL
from sqlglot import exp

from tests.query_optimizations.utils import TimedTestBase
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
from blendsql.parse import SubqueryContextManager
from blendsql.parse.dialect import (
    _parse_one,
    BlendSQLDuckDB,
    BlendSQLSQLite,
    BlendSQLPostgres,
    BlendSQLDialect,
)


@pytest.fixture(scope="session")
def bsql() -> BlendSQL:
    return BlendSQL(
        {
            "customers": pd.DataFrame(
                {
                    "customer_id": [1, 2, 3, 4, 5],
                    "name": [
                        "Alice Chen",
                        "Bob Martinez",
                        "Carol Smith",
                        "David Kim",
                        "Alice Chen",
                    ],
                    "email": [
                        "alice@email.com",
                        "bob@email.com",
                        "carol@email.com",
                        "david@email.com",
                        "emma@email.com",
                    ],
                    "signup_date": pd.to_datetime(
                        [
                            "2023-03-15",
                            "2023-06-22",
                            "2023-09-01",
                            "2024-01-10",
                            "2024-02-28",
                        ]
                    ).date,
                    "country": ["Chile", "Mexico", "UK", "US", "Canada"],
                }
            ),
            "orders": pd.DataFrame(
                {
                    "order_id": [101, 102, 103, 104, 105, 106, 107, 108],
                    "customer_id": [1, 1, 2, 3, 3, 3, 4, 5],
                    "order_date": pd.to_datetime(
                        [
                            "2023-04-01",
                            "2023-07-15",
                            "2023-08-10",
                            "2023-10-05",
                            "2023-12-20",
                            "2024-02-14",
                            "2024-02-01",
                            "2024-03-10",
                        ]
                    ).date,
                    "total_amount": [
                        150.00,
                        89.50,
                        320.00,
                        45.99,
                        210.75,
                        67.50,
                        599.99,
                        125.00,
                    ],
                    "status": [
                        "completed",
                        "completed",
                        "completed",
                        "completed",
                        "completed",
                        "shipped",
                        "shipped",
                        "pending",
                    ],
                }
            ),
        },
        ingredients={
            test_starts_with,
            get_length,
            select_first_sorted,
            do_join,
            return_aapl,
            get_table_size,
            select_first_option,
            return_aapl_alias,
        },
    )


class TestEarlyExitOperations(TimedTestBase):
    def test_basic_filter_early_exit(self, bsql):
        """
        Since the whole 'get the first value that satisfies this condition' logic isn't deterministic, we check
        that the single returned name is in our list of viable candidates.
        """
        for limit_n in range(1, 3):
            smoothie = self.assert_blendsql_equals_sql(
                bsql,
                blendsql_query=f"""
                SELECT name FROM customers c
                JOIN orders o 
                ON c.customer_id = o.customer_id
                WHERE status = 'shipped'
                AND {{{{get_length(country)}}}} = 2
                LIMIT {limit_n}
                """,
                sql_query=f"""
                SELECT name FROM customers c
                JOIN orders o 
                ON c.customer_id = o.customer_id
                WHERE status = 'shipped'
                AND LENGTH(country) = 2
                LIMIT {limit_n}
                """,
                expected_num_values_passed=limit_n,
                skip_assert_equality=True,
            )
            assert list(smoothie.df.values.flat)[0] in ["Carol Smith", "David Kim"]

    def test_regex_search_early_exit(self, bsql):
        smoothie = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT name FROM customers c
            JOIN orders o 
            ON c.customer_id = o.customer_id
            WHERE name LIKE 'A%'
            AND order_id != 102
            AND {{test_starts_with('C', country)}} = TRUE
            LIMIT 1
            """,
            sql_query="""
            SELECT name FROM customers c
            JOIN orders o 
            ON c.customer_id = o.customer_id
            WHERE name LIKE 'A%'
            AND order_id != 102
            AND country LIKE 'C%' = TRUE
            LIMIT 1
            """,
            expected_num_values_passed=1,
            args=["C"],
            skip_assert_equality=True,
        )
        # Both candidate rows have the same name
        assert list(smoothie.df.values.flat)[0] == "Alice Chen"

    def test_no_early_exit(self, bsql):
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT * FROM customers c
            JOIN orders o
            ON c.customer_id = o.customer_id
            WHERE (
                o.total_amount > 100
                AND {{test_starts_with('A', c.name)}}
            ) OR c.name = 'Bob Martinez'
            LIMIT 1
            """,
            sql_query="""
            SELECT * FROM customers c
            JOIN orders o
            ON c.customer_id = o.customer_id
            WHERE (
                o.total_amount > 100
                AND c.name LIKE 'A%'
            ) OR c.name = 'Bob Martinez'
            LIMIT 1
            """,
            args=["A"],
        )

    def test_early_exit_with_function_alias(self, bsql):
        """744c696"""
        for enable_early_exit, expected_num_values_passed in [(True, 1), (False, 2)]:
            smoothie = self.assert_blendsql_equals_sql(
                bsql,
                blendsql_query="""
                SELECT name,
                {{test_starts_with('C', country)}} AS aliased_function
                FROM customers c
                JOIN orders o 
                ON c.customer_id = o.customer_id
                WHERE name LIKE 'A%'
                AND order_id != 102
                AND aliased_function = TRUE
                LIMIT 1
                """,
                sql_query="""
                SELECT name,
                Country LIKE 'C%' AS aliased_function
                FROM customers c
                JOIN orders o 
                ON c.customer_id = o.customer_id
                WHERE name LIKE 'A%'
                AND order_id != 102
                AND aliased_function = TRUE
                LIMIT 1
                """,
                expected_num_values_passed=expected_num_values_passed,
                args=["C"],
                skip_assert_equality=True,
                enable_early_exit=enable_early_exit,
            )
            # Both candidate rows have the same name
            assert list(smoothie.df.values.flat)[0] == "Alice Chen"

    def test_eligible_for_cascade_filter(self):
        def get_scm(query: str, dialect: BlendSQLDialect) -> SubqueryContextManager:
            return SubqueryContextManager(
                dialect=dialect,
                node=_parse_one(
                    query,
                    dialect=dialect,
                ),
                prev_subquery_has_ingredient=False,
                ingredient_alias_to_parsed_dict={
                    "LLMMap": {"kwargs_dict": {}},
                },
            )

        for dialect in [BlendSQLDuckDB, BlendSQLSQLite, BlendSQLPostgres]:
            scm = get_scm(
                """
            SELECT merchant FROM transactions t
            WHERE {{LLMMap('This is a question', merchant)}} = TRUE
            LIMIT 1
            """,
                dialect,
            )
            function_node = scm.node.find(exp.BlendSQLFunction)
            assert scm.get_exit_condition(function_node) is not None

            # 7b26714
            scm = get_scm(
                """
              SELECT merchant
              FROM transactions t
              WHERE {{LLMMap('This is a question', merchant)}} = TRUE
              AND (t.name = 'Betsy' OR t.year = 2025)
              LIMIT 1
              """,
                dialect,
            )
            function_node = scm.node.find(exp.BlendSQLFunction)
            assert scm.get_exit_condition(function_node) is not None
