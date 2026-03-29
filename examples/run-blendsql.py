from dotenv import load_dotenv

from blendsql import BlendSQL, GLOBAL_HISTORY
from blendsql.models import VLLM
from blendsql.common.utils import fetch_from_hub

load_dotenv()
"""
vllm serve cyankiwi/Qwen3.5-9B-AWQ-4bit --host 0.0.0.0 \
--port 8000 \
--enable-prefix-caching \
--max-model-len 8000 \
--structured-outputs-config.backend guidance \
--language-model-only \
--reasoning-parser qwen3 \
--gpu_memory_utilization 0.8 \
--max-cudagraph-capture-size 32 \
--enable-prompt-tokens-details
"""

# Prepare our BlendSQL connection
bsql = BlendSQL(
    # DuckDB(con=duckdb.connect("./research/movies_sembench/src/data/movie_database_2000.duckdb")),
    fetch_from_hub("ecomm/sf_2000/ecomm_database_2000.duckdb"),
    model=VLLM(model_name_or_path="RedHatAI/gemma-3-12b-it-quantized.w4a16"),
    verbose=True,
)

# print(f'{bsql.execute("SELECT COUNT(*) FROM reviews").df().values.item():,} total rows')

# If we've already processed this tablename before in the same subquery,
# it's a self-join, or something similar where a single table is used as a reference
# with more than one referent. As a result, our base 'swap in temp table strings by
# finding the original table name in the query' strategy doesn't work.
# We need to use the aliasname to identify the temp table we create.
# r1.reviewText, r2.reviewText

smoothie = bsql.execute(
    """
SELECT
  i.id as id,
  i.link,
  {{LLMMap('What is the primary color of the product in this image? Only return the base color, nothing else.', i.link)}}
FROM styles_details s
JOIN image_mapping i on s.id = i.id
WHERE s.baseColour IN ('Black', 'Blue', 'Red', 'White', 'Orange', 'Green')
LIMIT 5
"""
)
smoothie.print_summary()
print(GLOBAL_HISTORY[-1])
exit()

smoothie = bsql.execute(
    """
WITH self_joined_reviews AS (
    SELECT DISTINCT
    r1.originalScore AS originalScore1,
    r2.originalScore AS originalScore2,
    r1.id as id1,
    r1.reviewId as reviewId1,
    r2.reviewId as reviewId2
    FROM Reviews r1
    JOIN Reviews r2
    ON r1.id = r2.id
    AND r1.reviewId < r2.reviewId
    WHERE r1.id = 'ant_man_and_the_wasp_quantumania'
    AND r2.id = 'ant_man_and_the_wasp_quantumania'
    AND originalScore1 IS NOT NULL AND originalScore1 LIKE '%/%' AND CAST(split_part(originalScore1, '/', 1) AS FLOAT) / CAST(split_part(originalScore1, '/', 2) AS FLOAT) <> 0.5
    AND originalScore2 IS NOT NULL AND originalScore2 LIKE '%/%' AND CAST(split_part(originalScore2, '/', 1) AS FLOAT) / CAST(split_part(originalScore2, '/', 2) AS FLOAT) <> 0.5
    ORDER BY r1.reviewId, r2.reviewId
) SELECT id1, reviewId1, reviewId2 FROM self_joined_reviews
WHERE {{
    LLMMap(
        'Is one score greater than 1/2 and the other less than 1/2?',
        originalScore1,
        originalScore2,
        context=(SELECT * FROM Reviews LIMIT 3)
    )
}} = TRUE
    """,
)
smoothie.print_summary()
print(GLOBAL_HISTORY[-1])
exit()
# ┌───────────────────┬───────────────────────────────────────────────────────┐
# │ Name              │ Known_For                                             │
# ├───────────────────┼───────────────────────────────────────────────────────┤
# │ George Washington │ Established federal government, First U.S. Preside... │
# │ John Adams │ XYZ Affair, Alien and Sedition Acts                   │
# │ Thomas Jefferson  │ Louisiana Purchase, Declaration of Independence       │
# └───────────────────┴───────────────────────────────────────────────────────┘
# ┌────────────┬──────────────────────┬─────────────────┬─────────────────────┐
# │   Time (s) │   # Generation Calls │   Prompt Tokens │   Completion Tokens │
# ├────────────┼──────────────────────┼─────────────────┼─────────────────────┤
# │    1.25158 │                    1 │             296 │                  16 │
# └────────────┴──────────────────────┴─────────────────┴─────────────────────┘

smoothie = bsql.execute(
    """
    SELECT * FROM People P
    WHERE P.Name IN {{
        LLMQA('First 3 presidents of the U.S?', quantifier='{3}')
    }}
    """,
    infer_gen_constraints=True,
)

smoothie.print_summary()
# ┌───────────────────┬───────────────────────────────────────────────────────┐
# │ Name              │ Known_For                                             │
# ├───────────────────┼───────────────────────────────────────────────────────┤
# │ George Washington │ Established federal government, First U.S. Preside... │
# │ John Adams        │ XYZ Affair, Alien and Sedition Acts                   │
# │ Thomas Jefferson  │ Louisiana Purchase, Declaration of Independence       │
# └───────────────────┴───────────────────────────────────────────────────────┘
# ┌────────────┬──────────────────────┬─────────────────┬─────────────────────┐
# │   Time (s) │   # Generation Calls │   Prompt Tokens │   Completion Tokens │
# ├────────────┼──────────────────────┼─────────────────┼─────────────────────┤
# │    1.25158 │                    1 │             296 │                  16 │
# └────────────┴──────────────────────┴─────────────────┴─────────────────────┘


smoothie = bsql.execute(
    """
    SELECT GROUP_CONCAT(Name, ', ') AS 'Names',
    {{
        LLMMap(
            'In which time period was this person born?',
            Name,
            options=Eras.Years
        )
    }} AS Born
    FROM People
    GROUP BY Born
    """,
)

smoothie.print_summary()
# ┌───────────────────────────────────────────────────────┬───────────┐
# │ Names                                                 │ Born      │
# ├───────────────────────────────────────────────────────┼───────────┤
# │ George Washington, John Adams, Thomas Jefferson, J... │ 1700-1800 │
# │ Sabrina Carpenter, Charli XCX, Elon Musk, Michelle... │ 2000-Now  │
# │ Elvis Presley                                         │ 1900-2000 │
# └───────────────────────────────────────────────────────┴───────────┘
# ┌────────────┬──────────────────────┬─────────────────┬─────────────────────┐
# │   Time (s) │   # Generation Calls │   Prompt Tokens │   Completion Tokens │
# ├────────────┼──────────────────────┼─────────────────┼─────────────────────┤
# │    1.03858 │                    2 │             544 │                  75 │
# └────────────┴──────────────────────┴─────────────────┴─────────────────────┘

smoothie = bsql.execute(
    """
SELECT {{
    LLMQA(
        'Describe BlendSQL in 50 words.',
        context=(
            SELECT content[0:5000] AS "README"
            FROM read_text('https://raw.githubusercontent.com/parkervg/blendsql/main/README.md')
        )
    )
}} AS answer
"""
)

smoothie.print_summary()
# ┌─────────────────────────────────────────────────────┐
# │ answer                                              │
# ├─────────────────────────────────────────────────────┤
# │ BlendSQL is a Python library that combines SQL a... │
# └─────────────────────────────────────────────────────┘

# ┌────────────┬──────────────────────┬─────────────────┬─────────────────────┐
# │   Time (s) │   # Generation Calls │   Prompt Tokens │   Completion Tokens │
# ├────────────┼──────────────────────┼─────────────────┼─────────────────────┤
# │    4.07617 │                    1 │            1921 │                  50 │
# └────────────┴──────────────────────┴─────────────────┴─────────────────────┘
