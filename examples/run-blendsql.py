import pandas as pd

import blendsql
from blendsql.ingredients import LLMMap, LLMQA
from blendsql.db import Pandas
from blendsql.models import LiteLLM

# Optionally set how many async calls to allow concurrently
# This depends on your OpenAI/Anthropic/etc. rate limits
blendsql.config.set_async_limit(10)

# Load model
model = LiteLLM("openai/gpt-4o-mini")  # requires .env file with `OPENAI_API_KEY`
# model = LiteLLM("anthropic/claude-3-haiku-20240307") # requires .env file with `ANTHROPIC_API_KEY`
# model = TransformersLLM('Qwen/Qwen1.5-0.5B') # run with any local Transformers model

# Prepare our local database
db = Pandas(
    {
        "People": pd.DataFrame(
            {
                "Name": [
                    "George Washington",
                    "John Quincy Adams",
                    "Thomas Jefferson",
                    "James Madison",
                    "James Monroe",
                    "Alexander Hamilton",
                    "Sabrina Carpenter",
                    "Charli XCX",
                    "Elon Musk",
                    "Michelle Obama",
                    "Elvis Presley",
                ],
                "Known_For": [
                    "Established federal government, First U.S. President",
                    "XYZ Affair, Alien and Sedition Acts",
                    "Louisiana Purchase, Declaration of Independence",
                    "War of 1812, Constitution",
                    "Monroe Doctrine, Missouri Compromise",
                    "Created national bank, Federalist Papers",
                    "Nonsense, Emails I Cant Send, Mean Girls musical",
                    "Crash, How Im Feeling Now, Boom Clap",
                    "Tesla, SpaceX, Twitter/X acquisition",
                    "Lets Move campaign, Becoming memoir",
                    "14 Grammys, King of Rock n Roll",
                ],
            }
        ),
        "Eras": pd.DataFrame({"Years": ["1800-1900", "1900-2000", "2000-Now"]}),
    }
)

# Write BlendSQL query
query = """
WITH Musicians AS
    (
        SELECT Name FROM People
        WHERE {{LLMMap('Is a singer?', 'People::Name')}} = TRUE
    )
SELECT Name AS "working late cuz they're a singer" FROM Musicians M
WHERE M.Name = {{LLMQA('Who wrote the song "Espresso?"')}}
"""
smoothie = blendsql.blend(
    query=query,
    db=db,
    ingredients={LLMMap, LLMQA},
    default_model=model,
    # Optional args below
    infer_gen_constraints=True,
    verbose=True,
)
print(smoothie.df)
# ┌─────────────────────────────────────┐
# │ working late cuz they're a singer   │
# ├─────────────────────────────────────┤
# │ Sabrina Carpenter                   │
# └─────────────────────────────────────┘
print(smoothie.summary())
# ┌────────────┬──────────────────────┬─────────────────┬─────────────────────┐
# │   Time (s) │   # Generation Calls │   Prompt Tokens │   Completion Tokens │
# ├────────────┼──────────────────────┼─────────────────┼─────────────────────┤
# │    0.12474 │                    1 │            1918 │                  42 │
# └────────────┴──────────────────────┴─────────────────┴─────────────────────┘
