import pandas as pd
import psutil

# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from blendsql import BlendSQL, GLOBAL_HISTORY
from blendsql.db import DuckDB
from blendsql.models import LiteLLM, LlamaCpp
from blendsql.common.utils import fetch_from_hub


USE_LOCAL_CONSTRAINED_MODEL = True

# Load model, either a local transformers model, or remote provider via LiteLLM
model = None
if True:
    if USE_LOCAL_CONSTRAINED_MODEL:
        model = LlamaCpp(
            model_name_or_path="unsloth/gemma-3-4b-it-GGUF",
            filename="gemma-3-4b-it-Q4_K_M.gguf",
            # model_name_or_path="bartowski/SmolLM2-135M-Instruct-GGUF",
            # filename="SmolLM2-135M-Instruct-Q4_K_M.gguf",
            config={
                "n_gpu_layers": -1,
                "n_ctx": 1028,
                "seed": 100,
                "n_threads": psutil.cpu_count(logical=False),
            },
            caching=False,
        )
        _ = model.model_obj
    else:
        model = LiteLLM("openai/gpt-4o-mini")

# Prepare our BlendSQL connection
bsql = BlendSQL(
    DuckDB.from_pandas(
        {
            "Movies": pd.read_csv(fetch_from_hub("movie/sf_2000/Movies.csv")),
            "Reviews": pd.read_csv(fetch_from_hub("movie/sf_2000/Reviews.csv")),
        },
    ),
    enable_early_exit=False,
    model=model,
    verbose=True,
)

print(f'{bsql.execute("SELECT COUNT(*) FROM reviews").df.values.item():,} total rows')

# If we've already processed this tablename before in the same subquery,
# it's a self-join, or something similar where a single table is used as a reference
# with more than one referent. As a result, our base 'swap in temp table strings by
# finding the original table name in the query' strategy doesn't work.
# We need to use the aliasname to identify the temp table we create.
# r1.reviewText, r2.reviewText
smoothie = bsql.execute(
    """
    SELECT reviewId, reviewText,
    {{
        LLMMap(
            'What is the sentiment of this review?',
            reviewText,
            options=('POSITIVE', 'NEGATIVE')
        )
    }} AS prediction,
    scoreSentiment AS reference
    FROM Reviews
    WHERE id = 'taken_3'
    AND prediction = 'POSITIVE'
    LIMIT 5;
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
