from blendsql import blend
from blendsql.db import SQLite
from blendsql.utils import tabulate, fetch_from_hub
from dotenv import load_dotenv
from tests.utils import (
    starts_with,
    get_length,
    select_first_sorted,
    do_join,
    return_aapl,
    get_table_size,
)

load_dotenv()
if __name__ == "__main__":
    blendsql = """
    SELECT Symbol FROM (
        SELECT DISTINCT Symbol FROM portfolio WHERE Symbol IN (
            SELECT Symbol FROM portfolio WHERE Quantity > 200
        )
    ) AS w WHERE {{starts_with('F', 'w::Symbol')}} = TRUE AND LENGTH(w.Symbol) > 3
    """
    # db = PostgreSQL("localhost:5432/mydb")
    db = SQLite(fetch_from_hub("multi_table.db"))
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients={
            starts_with,
            get_length,
            select_first_sorted,
            do_join,
            return_aapl,
            get_table_size,
        },
        verbose=True,
        # blender=TransformersLLM("Qwen/Qwen1.5-0.5B"),
        schema_qualify=False,
    )
    print("--------------------------------------------------")
    print("ANSWER:")
    print(tabulate(smoothie.df))
    print("--------------------------------------------------")
    print(smoothie.meta.num_values_passed)
    print(smoothie.meta.prompts)
