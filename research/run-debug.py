from blendsql import blend, LLMJoin
from blendsql.db import SQLite, PostgreSQL
from blendsql.models import OpenaiLLM
from blendsql.utils import fetch_from_hub, tabulate

if __name__ == "__main__":
    blendsql = """
    SELECT date, rival, score, documents.content AS "Team Description" FROM w
    JOIN {{
        LLMJoin(
            left_on='documents::title',
            right_on='w::rival'
        )
    }} WHERE rival = 'nsw waratahs'
    """
    # Make our smoothie - the executed BlendSQL script
    smoothie = blend(
        query=blendsql,
        db=SQLite(
            fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db")
        ),
        blender=OpenaiLLM("gpt-3.5-turbo"),
        ingredients={LLMJoin},
    )
    print(tabulate(smoothie.df))

    smoothie = blend(
        query=blendsql,
        db=PostgreSQL(
            "blendsql@localhost:5432/1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1"
        ),
        blender=OpenaiLLM("gpt-3.5-turbo"),
        ingredients={LLMJoin},
    )
    print(tabulate(smoothie.df))
