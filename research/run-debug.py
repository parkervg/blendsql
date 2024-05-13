from blendsql import blend
from blendsql.db import SQLite
from blendsql.utils import fetch_from_hub
from blendsql.utils import tabulate
from dotenv import load_dotenv
from tests.utils import (
    starts_with,
    get_length,
    select_first_sorted,
    get_table_size,
)

load_dotenv()
if __name__ == "__main__":
    blendsql = """
           SELECT * FROM cars
            WHERE {{LLMMap('Was this model made in the 21st century?', 'cars::year')}} = TRUE
    """
    blendsql = """
        SELECT NOT EXISTS (
        SELECT * FROM transactions WHERE {{get_length('length', 'transactions::merchant')}} > 4 AND amount > 500
    ) OR (
        {{
            get_table_size('Table size?', (select * from transactions where amount < 500))
        }}
    )
    """
    # db = PostreSQL("localhost:5432/mydb")
    db = SQLite(fetch_from_hub("single_table.db"))
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients={
            starts_with,
            get_length,
            select_first_sorted,
            get_table_size,
        },
        verbose=True,
    )
    print("--------------------------------------------------")
    print("ANSWER:")
    print(tabulate(smoothie.df))
    print("--------------------------------------------------")
    print(smoothie.meta.num_values_passed)
    print(smoothie.meta.prompts)
