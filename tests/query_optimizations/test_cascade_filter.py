import pytest
import pandas as pd
from blendsql import BlendSQL
from blendsql.parse import SubqueryContextManager
from blendsql.parse.dialect import (
    _parse_one,
    BlendSQLDuckDB,
    BlendSQLSQLite,
    BlendSQLPostgres,
    BlendSQLDialect,
)
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


class TestCascadeFilter(TimedTestBase):
    def test_basic_cascade_filter(self, bsql):
        expected_num_values_passed_with_filter: int = bsql.db.execute_to_list(
            """
            SELECT (
                SELECT COUNT(DISTINCT name) FROM customers
                WHERE customer_id > 2
            ) + (
                SELECT COUNT(DISTINCT country) FROM customers
                WHERE customer_id > 2
                AND name LIKE 'C%'
            )
            """,
            to_type=int,
        )[0]

        expected_num_values_passed_without_filter: int = (
            bsql.db.execute_to_list(
                """
            SELECT (
                SELECT COUNT(DISTINCT name) FROM customers
                WHERE customer_id > 2
            ) 
            """,
                to_type=int,
            )[0]
            * 2
        )

        for enable_cascade_filter, expected_num_values_passed in [
            (True, expected_num_values_passed_with_filter),
            (False, expected_num_values_passed_without_filter),
        ]:
            _ = self.assert_blendsql_equals_sql(
                bsql,
                blendsql_query="""
                SELECT country FROM customers
                WHERE {{test_starts_with('C', name)}} = TRUE
                AND {{get_length(country)}} = 2
                AND customer_id > 2
                """,
                sql_query="""
                SELECT country FROM customers
                WHERE customer_id > 2
                AND name LIKE 'C%'
                AND LENGTH(country) = 2
                """,
                expected_num_values_passed=expected_num_values_passed,
                enable_cascade_filter=enable_cascade_filter,
            )

    def test_do_not_cascade_filter(self, bsql):
        """If a BlendSQL function is impacted by an OR, do not apply cascade filter."""
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
            SELECT (
                SELECT COUNT(DISTINCT name) FROM customers
            ) + (
                SELECT COUNT(DISTINCT country) FROM customers
            )
            """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT country FROM customers
            WHERE customer_id > 2
            AND {{test_starts_with('C', name)}} = TRUE
            OR {{get_length(country)}} = 2
            """,
            sql_query="""
            SELECT country FROM customers
            WHERE customer_id > 2
            AND name LIKE 'C%'
            OR LENGTH(country) = 2
            """,
            expected_num_values_passed=expected_num_values_passed,
        )

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
            assert (
                get_scm(
                    """
                SELECT merchant FROM transactions t
                  WHERE {{LLMMap('This is a question', merchant)}} = TRUE
                  ORDER BY {{LLMMap('This is another question', merchant)}} LIMIT 1 OFFSET 2
                """,
                    dialect=dialect,
                ).is_eligible_for_cascade_filter()
                == False
            )

            assert (
                get_scm(
                    """
                SELECT merchant FROM transactions t
                  WHERE {{LLMMap('This is a question', merchant)}} = TRUE
                  AND {{LLMMap('This is another question', merchant)}} = FALSE
                  ORDER BY a_different_column LIMIT 1 OFFSET 2
                """,
                    dialect=dialect,
                ).is_eligible_for_cascade_filter()
                == True
            )
