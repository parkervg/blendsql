from blendsql import blend, LLMJoin, LLMMap, LLMQA, LLMValidate, ImageCaption
from blendsql.db import SQLite
from blendsql.utils import fetch_from_hub

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
    """
    SELECT * FROM w
      WHERE city = {{
          LLMQA(
              'Which city is located 120 miles west of Sydney?',
              (SELECT * FROM documents),
              options='w::city'
          )
      }}
    """,
    """
    SELECT date, rival, score, documents.content AS "Team Description" FROM w
    JOIN {{
        LLMJoin(
            left_on='documents::title',
            right_on='w::rival'
        )
    }}
    """,
    """
    {{
        LLMQA(
            'What is this table about?',
            (SELECT * FROM w;)
        )
    }}
    """,
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
    from blendsql.models import OpenaiLLM, TransformersVisionModel

    db = SQLite(fetch_from_hub("national_parks.db"))
    ingredients = {
        LLMQA,
        LLMMap,
        LLMJoin,
        LLMValidate,
        ImageCaption.from_args(model=TransformersVisionModel("Mozilla/distilvit")),
    }
    q = """
SELECT COUNT(*) AS "Count" FROM parks
    WHERE {{LLMMap('How many states?', 'parks::Location')}} > 1
    """
    smoothie = blend(
        query=q,
        db=db,
        default_model=OpenaiLLM("gpt-4", caching=False),
        # default_model=TransformersLLM("Qwen/Qwen1.5-0.5B", caching=False),
        # default_model=OllamaLLM("phi3", caching=False),
        verbose=True,
        ingredients=ingredients,
    )
    print(smoothie.df.to_markdown(index=False))
