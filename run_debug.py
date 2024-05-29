from blendsql import blend, LLMJoin, LLMMap, LLMQA
from blendsql.db import SQLite
from blendsql.models import TransformersLLM
from blendsql.utils import fetch_from_hub
from tqdm import tqdm

# TEST_QUERIES = [
#     """
#     SELECT DISTINCT venue FROM w
#       WHERE city = 'sydney' AND {{
#           LLMMap(
#               'More than 30 total points?',
#               'w::score'
#           )
#       }} = TRUE
#     """,
#     """
#     SELECT * FROM w
#       WHERE city = {{
#           LLMQA(
#               'Which city is located 120 miles west of Sydney?',
#               (SELECT * FROM documents WHERE documents MATCH 'sydney OR 120'),
#               options='w::city'
#           )
#       }}
#     """,
#     """
#     SELECT date, rival, score, documents.content AS "Team Description" FROM w
#     JOIN {{
#         LLMJoin(
#             left_on='documents::title',
#             right_on='w::rival'
#         )
#     }}
#     """
# ]

TEST_QUERIES = [
    """
    SELECT title, player FROM w JOIN {{
        LLMJoin(
            left_on='documents::title',
            right_on='w::player'
        )
    }}
    """
]
if __name__ == "__main__":
    """
    Without cached LLM response (10 runs):
        before: 3.16
        after: 1.91
    With cached LLM response (100 runs):
        before: 0.0175
        after: 0.0166
    With cached LLM response (30 runs):
        with fuzzy join: 0.431
        without fuzzy join: 0.073
    Without cached LLM response (30 runs):
        with fuzzy join: 0.286
        without fuzzy join: 318.85
    """
    db = SQLite(fetch_from_hub("1966_NBA_Expansion_Draft_0.db"))
    model = TransformersLLM("Qwen/Qwen1.5-0.5B", caching=False)
    times = []
    for i in range(30):
        for q in TEST_QUERIES:

            # Make our smoothie - the executed BlendSQL script
            smoothie = blend(
                query=q,
                db=db,
                blender=model,
                verbose=False,
                ingredients={LLMJoin, LLMMap, LLMQA},
            )
            times.append(smoothie.meta.process_time_seconds)
    print(smoothie.df)
    print(f"Average time across {len(times)} runs: {sum(times) / len(times)}")