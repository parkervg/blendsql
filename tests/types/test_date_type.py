import pytest
import pandas as pd

from blendsql import BlendSQL


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
                        "Emma Wilson",
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
                    "country": ["USA", "Mexico", "UK", "South Korea", "Canada"],
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
        }
    )


def test_date_type_infer(bsql, constrained_model):
    _ = bsql.execute(
        """
        SELECT * FROM customers c 
        JOIN orders o ON c.customer_id = o.customer_id
        WHERE status = 'completed'
        AND order_date > {{LLMQA('When was the first case of COVID-19?')}}
        """,
        model=constrained_model,
    )
