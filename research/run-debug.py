from blendsql import LLMMap, LLMQA, LLMJoin, blend
from blendsql.db import SQLiteDBConnector
from blendsql.llms import AzureOpenaiLLM

if __name__ == "__main__":
    blendsql = """
       {{
            LLMQA(
                'What is the middle name of this player?',
                (
                    SELECT documents.title AS 'Player', documents.content FROM documents
                    JOIN {{
                        LLMJoin(
                            left_on='w::player',
                            right_on='documents::title'
                        )
                    }} WHERE w."rank" = 2
                )
            )
        }}
       """
    db_path = "research/db/hybridqa/List_of_National_Football_League_rushing_yards_leaders_0.db"

    db = SQLiteDBConnector(db_path)
    smoothie = blend(
        query=blendsql,
        db=db,
        ingredients={LLMMap, LLMQA, LLMJoin},
        blender=AzureOpenaiLLM("gpt-4"),
        infer_gen_constraints=True,
        silence_db_exec_errors=False,
        verbose=True,
    )
    print("--------------------------------------------------")
    print("ANSWER:")
    print([i for i in smoothie.df.values.flat])
    print("--------------------------------------------------")
    print(smoothie.meta.example_map_outputs)
    print(smoothie.meta.num_values_passed)
