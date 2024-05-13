from blendsql import blend, LLMMap
from blendsql.db import PostgreSQL
from blendsql.models import TransformersLLM
from blendsql.utils import tabulate
from dotenv import load_dotenv

load_dotenv()
if __name__ == "__main__":
    blendsql = """
           SELECT * FROM cars
            WHERE {{LLMMap('Was this model made in the 21st century?', 'cars::year')}} = TRUE
    """
    db = PostgreSQL("localhost:5432/mydb")
    # db = SQLite(fetch_from_hub("single_table.db"))
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients={LLMMap},
        verbose=True,
        blender=TransformersLLM("Qwen/Qwen1.5-0.5B"),
        schema_qualify=False,
    )
    print("--------------------------------------------------")
    print("ANSWER:")
    print(tabulate(smoothie.df))
    print("--------------------------------------------------")
    print(smoothie.meta.num_values_passed)
    print(smoothie.meta.prompts)
