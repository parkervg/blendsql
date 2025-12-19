import pandas as pd
import psutil
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
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
            # model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            # filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
            model_name_or_path="bartowski/SmolLM2-135M-Instruct-GGUF",
            filename="SmolLM2-135M-Instruct-Q4_K_M.gguf",
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
            "Movies": pd.read_csv(fetch_from_hub("movie/rotten_tomatoes_movies.csv")),
            "Reviews": pd.read_csv(
                fetch_from_hub("movie/rotten_tomatoes_movie_reviews.csv")
            ),
        }
    ),
    # {
    #     "People": pd.DataFrame(
    #         {
    #             "Name": [
    #                 "George Washington",
    #                 "John Adams",
    #                 "Thomas Jefferson",
    #                 "James Madison",
    #                 "James Monroe",
    #                 "Alexander Hamilton",
    #                 "Sabrina Carpenter",
    #                 "Charli XCX",
    #                 "Elon Musk",
    #                 "Michelle Obama",
    #                 "Elvis Presley",
    #             ],
    #             "Known_For": [
    #                 "Established federal government, First U.S. President",
    #                 "XYZ Affair, Alien and Sedition Acts",
    #                 "Louisiana Purchase, Declaration of Independence",
    #                 "War of 1812, Constitution",
    #                 "Monroe Doctrine, Missouri Compromise",
    #                 "Created national bank, Federalist Papers",
    #                 "Nonsense, Emails I Cant Send, Mean Girls musical",
    #                 "Crash, How Im Feeling Now, Boom Clap",
    #                 "Tesla, SpaceX, Twitter/X acquisition",
    #                 "Lets Move campaign, Becoming memoir",
    #                 "14 Grammys, King of Rock n Roll",
    #             ],
    #         }
    #     ),
    #     "Eras": pd.DataFrame({"Years": ["1700-1800", "1800-1900", "1900-2000", "2000-Now"]}),
    # },
    model=model,
    verbose=True,
)
#
# smoothie = bsql.execute(
#     """
#     SELECT {{LLMMap('What is their name?', Name)}} FROM People
#     """
# )
# smoothie.print_summary()
# print(GLOBAL_HISTORY[-1])
# exit()


print(f'{bsql.execute("SELECT COUNT(*) FROM reviews").df.values.item():,} total rows')

# If we've already processed this tablename before in the same subquery,
# it's a self-join, or something similar where a single table is used as a reference
# with more than one referent. As a result, our base 'swap in temp table strings by
# finding the original table name in the query' strategy doesn't work.
# We need to use the aliasname to identify the temp table we create.
# r1.reviewText, r2.reviewText
smoothie = bsql.execute(
    """
    SELECT reviewId
    FROM Reviews
    WHERE {{
        LLMMap('Is the movie review clearly positive?', reviewText)
    }} = TRUE
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
