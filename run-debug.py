from blendsql import blend, LLMJoin, LLMMap, LLMQA
from blendsql.db import SQLite
from blendsql.utils import fetch_from_hub

# db = DuckDB.from_pandas(
#     pd.DataFrame(
#             {
#                 "name": ["John", "Parker"],
#                 "age": [12, 26]
#             },
#     )
# )
# DuckDB.from_sqlite(fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db"))
# print()
#
# db.to_temp_table(df=pd.DataFrame(
#     {
#         "class": ["Boxing 101"],
#         "num_enrolled": [23]
#     }
# ), tablename="classes"
# )
# print()


TEST_QUERIES = [
    """
    SELECT DISTINCT venue FROM w
      WHERE city = 'sydney' AND {{
          LLMMap(
              'More than 30 total points?',
              'w::score'
          )
      }} = TRUE
    """,
    # """
    # SELECT * FROM w
    #   WHERE city = {{
    #       LLMQA(
    #           'Which city is located 120 miles west of Sydney?',
    #           (SELECT * FROM documents),
    #           options='w::city'
    #       )
    #   }}
    # """,
    # """
    # SELECT date, rival, score, documents.content AS "Team Description" FROM w
    # JOIN {{
    #     LLMJoin(
    #         left_on='documents::title',
    #         right_on='w::rival'
    #     )
    # }}
    # """,
    # """
    # {{
    #     LLMQA(
    #         'What is this table about?',
    #         (SELECT * FROM w;)
    #     )
    # }}
    # """
]

# TEST_QUERIES = [
#     """
#     SELECT title, player FROM w JOIN {{
#         LLMJoin(
#             left_on='documents::title',
#             right_on='w::player'
#         )
#     }} WHERE {{
#         LLMMap(
#            'How many years with the franchise?',
#            'w::career with the franchise'
#         )
#     }} > 5
#     """
# ]
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

    DuckDB, with iterating over temp_tables (10 runs):
        0.5055
    Recreating database on reset:
        0.538 - 0.7

    """
    db = SQLite(
        fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db")
    )
    ingredients = {LLMQA, LLMMap, LLMJoin}
    # db = SQLite(fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db"))
    from blendsql.models import OpenaiLLM

    # model = OpenaiLLM("gpt-3.5-turbo", caching=False)
    times = []
    for _i in range(1):
        for q in TEST_QUERIES:
            # Make our smoothie - the executed BlendSQL script
            smoothie = blend(
                query=q,
                db=db,
                blender=OpenaiLLM("gpt-3.5-turbo", caching=False),
                # blender=TransformersLLM("microsoft/Phi-3-mini-128k-instruct"),
                # blender=OllamaLLM("phi3", caching=False),
                verbose=True,
                ingredients={LLMJoin.from_args(use_skrub_joiner=False), LLMMap, LLMQA},
            )
            times.append(smoothie.meta.process_time_seconds)
    # print(smoothie.df)
    print(f"Average time across {len(times)} runs: {sum(times) / len(times)}")
