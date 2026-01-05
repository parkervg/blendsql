import pytest
import pandas as pd
from blendsql import BlendSQL
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
                        "Amy Adams",
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


class TestGeneralOptimizations(TimedTestBase):
    def test_early_exit_with_cascade(self, bsql):
        """b6a50ef
        Both customers from the country that starts with 'C' have 'A' as the first letter in their name.

        We expect both the early exit and cascade filter to kick in on the final map ingredient.
        """
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
            SELECT (
                SELECT COUNT(DISTINCT country) FROM customers
            ) + 1
            """,
            to_type=int,
        )[0]
        smoothie = bsql.execute(
            """
            SELECT customer_id FROM customers
            WHERE {{test_starts_with('C', country)}} = TRUE
            AND {{test_starts_with('A', name)}} = TRUE
            LIMIT 1
            """
        )
        assert list(smoothie.df.values.flat)[0] in [1, 5]
        assert smoothie.meta.num_values_passed == expected_num_values_passed

    def test_early_exit_with_cascade_and_alias(self, bsql):
        """
        f8302b6
        """
        bsql_query = """
        SELECT 
        customer_id, 
        {{test_starts_with('C', country)}} AS startsWithC
        FROM customers
        WHERE startsWithC
        AND {{test_starts_with('A', name)}}
        """
        expected_num_values_passed: int = bsql.db.execute_to_list(
            """
            SELECT (
                SELECT COUNT(DISTINCT country) FROM customers
            ) + 1
            """,
            to_type=int,
        )[0]
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query=bsql_query,
            sql_query="""
            SELECT 
            customer_id, 
            country LIKE 'C%' AS startsWithC
            FROM customers
            WHERE startsWithC
            AND name LIKE 'A%'
          """,
        )
        smoothie = bsql.execute(bsql_query + " LIMIT 1")
        assert list(smoothie.df.values.flat)[0] in [1, 5]
        assert smoothie.meta.num_values_passed == expected_num_values_passed
