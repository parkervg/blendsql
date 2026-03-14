import pandas as pd

from blendsql import BlendSQL, GLOBAL_HISTORY
from blendsql.models import VLLM

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
    {
        "w": pd.DataFrame(
            {
                "player_name": ["John Wall", "Jayson Tatum"],
                "Report": ["He had 2 assists", "He only had 1 assist"],
                "AnotherOptionalReport": ["He had 26pts", "He scored 51pts!"],
            }
        ),
        "v": pd.DataFrame({"people": ["john", "jayson", "emily"]}),
        "names_and_ages": pd.DataFrame(
            {
                "Name": ["Tommy", "Sarah", "Tommy"],
                "Description": ["He is 24 years old", "She's 12", "He's only 3"],
            }
        ),
        "movie_reviews": pd.DataFrame(
            {"review": ["I love this movie!", "This was SO GOOD"]}
        ),
    },
    model=VLLM(
        model_name_or_path="RedHatAI/gemma-3-12b-it-quantized.w4a16",
        base_url="http://127.0.0.1:8000/v1/",
    ),
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
SELECT *, {{
            LLMMap(
                'Is {} a boy?',
                Name,
                Description,
                options=('yes', 'no'),
                context=(SELECT 'parker')
            )
        }} FROM "names_and_ages"
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
