import pytest
from blendsql import BlendSQL
from blendsql.common.utils import fetch_from_hub
from blendsql.db import DuckDB, SQLite
from tests.utils import return_true
from tests.query_optimizations.utils import TimedTestBase


bsql_connections = [
    BlendSQL(
        SQLite(fetch_from_hub("multi_table.db")),
        ingredients={return_true},
        infer_gen_constraints=False,
    ),
    BlendSQL(
        DuckDB.from_sqlite(fetch_from_hub("multi_table.db")),
        ingredients={return_true},
        infer_gen_constraints=False,
    ),
]


class TestOverhead(TimedTestBase):
    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_1(self, bsql: BlendSQL):
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT Symbol, "North America", "Japan" FROM geographic
                WHERE geographic.Symbol IN (
                    SELECT Symbol FROM portfolio
                    WHERE {{return_true()}} 
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
                    WHERE TRUE
                    AND Symbol in (
                        SELECT Symbol FROM constituents
                        WHERE constituents.Sector = 'Information Technology'
                    )
                )
            """,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_2(self, bsql: BlendSQL):
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT DISTINCT Name FROM constituents
            JOIN account_history ON constituents.Symbol = account_history.Symbol
            JOIN portfolio on constituents.Symbol = portfolio.Symbol
            WHERE account_history."Run Date" > '2021-02-23'
            AND {{return_true()}}
            AND portfolio.Symbol IS NOT NULL
            ORDER BY LENGTH(constituents.Name), constituents.Name LIMIT 4
            """,
            sql_query="""
            SELECT DISTINCT Name FROM constituents
            JOIN account_history ON constituents.Symbol = account_history.Symbol
            JOIN portfolio on constituents.Symbol = portfolio.Symbol
            WHERE account_history."Run Date" > '2021-02-23'
            AND TRUE
            AND portfolio.Symbol IS NOT NULL
            ORDER BY LENGTH(constituents.Name), constituents.Name LIMIT 4
            """,
        )

    @pytest.mark.parametrize("bsql", bsql_connections)
    def test_3(self, bsql: BlendSQL):
        _ = self.assert_blendsql_equals_sql(
            bsql,
            blendsql_query="""
            SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
                FROM account_history
                JOIN constituents c ON account_history.Symbol = c.Symbol
                WHERE c.Sector = 'Information Technology'
                AND {{return_true()}}
                AND LOWER(account_history.Action) like '%dividend%'
                ORDER BY "Total Dividend Payout ($$)"
            """,
            sql_query="""
            SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
                FROM account_history
                JOIN constituents ON account_history.Symbol = constituents.Symbol
                WHERE constituents.Sector = 'Information Technology'
                AND TRUE
                AND LOWER(account_history.Action) like '%dividend%'
                ORDER BY "Total Dividend Payout ($$)"
            """,
            args=["A"],
        )
