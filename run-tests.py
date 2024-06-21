from blendsql import blend
from blendsql.db import DuckDB, SQLite
from blendsql.utils import fetch_from_hub
from tests.utils import (
    starts_with,
    get_length,
    select_first_sorted,
    get_table_size,
    select_first_option,
    do_join,
)

if __name__ == "__main__":
    query = """
    {{
        select_first_option(
            'I hope this test works',
            (SELECT * FROM transactions),
            options=(SELECT DISTINCT merchant FROM transactions WHERE merchant = 'Paypal')
        )
    }}
    """
    ingredients = {
        starts_with,
        get_length,
        select_first_sorted,
        get_table_size,
        select_first_option,
        do_join,
    }
    sqlite_db = SQLite(fetch_from_hub("single_table.db"))
    db = DuckDB.from_sqlite(fetch_from_hub("single_table.db"))

    smoothie = blend(query=query, db=db, ingredients=ingredients, verbose=True)
    # sql = """
    # SELECT w."Percent of Account" FROM (SELECT * FROM "portfolio" WHERE Quantity > 200 OR "Today''s Gain/Loss Percent" > 0.05) as w
    # JOIN geographic ON w.Symbol = geographic.Symbol
    # WHERE w.Symbol LIKE 'F%'
    # AND w."Percent of Account" < 0.2
    #    """
    # sql_df = db.execute_to_df(sql)
    from tests.utils import assert_equality

    assert_equality(smoothie=smoothie, sql_df=sql_df, args=["Z"])
    print()
