from blendsql import LLMMap, LLMQA, LLMJoin, blend
from blendsql.db import SQLite
from blendsql.models import OpenaiLLM
from blendsql.utils import fetch_from_hub
from blendsql.utils import tabulate
from dotenv import load_dotenv

load_dotenv()
if __name__ == "__main__":
    # blendsql = """
    #    SELECT * FROM w
    #     WHERE city = {{
    #         LLMQA(
    #             'Which city is located 120 miles west of Sydney?',
    #             (SELECT * FROM documents WHERE documents MATCH 'sydney OR 120'),
    #             options='w::city'
    #         )
    #     }}
    #    """
    blendsql = """
           SELECT * FROM cars
            WHERE {{LLMMap('Was this model made in the 21st century?', 'cars::year')}} = TRUE       
    """
    # db = PostreSQL("localhost:5432/mydb")
    db = SQLite(
        fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db")
    )
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients={LLMMap, LLMQA, LLMJoin},
        blender=OpenaiLLM("gpt-4"),
        infer_gen_constraints=True,
        verbose=True,
        schema_qualify=False,
    )
    print("--------------------------------------------------")
    print("ANSWER:")
    print(tabulate(smoothie.df))
    print("--------------------------------------------------")
    print(smoothie.meta.example_map_outputs)
    print(smoothie.meta.num_values_passed)
