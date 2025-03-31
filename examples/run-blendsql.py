import pandas as pd
from guidance.chat import ChatMLTemplate

from blendsql import config, BlendSQL
from blendsql.ingredients import LLMMap, LLMQA
from blendsql.models import TransformersLLM
from blendsql.ingredients import LLMJoin


# Optionally set how many async calls to allow concurrently
# This depends on your OpenAI/Anthropic/etc. rate limits
config.set_async_limit(10)

# Load model
# model = TransformersLLM("microsoft/Phi-3.5-mini-instruct", config={"device_map": "auto"}, caching=False) # run with any local Transformers model
# model = TransformersLLM("meta-llama/Llama-3.1-8B-Instruct", config={"device_map": "auto"}, caching=False) # run with any local Transformers model

# Prepare our local database
bsql = BlendSQL(
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
    },
    ingredients={LLMMap, LLMQA, LLMJoin},
    model=TransformersLLM(
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        config={"chat_template": ChatMLTemplate, "device_map": "cpu"},
        caching=False,
    ),
)

smoothie = bsql.execute(
    """
    SELECT * FROM People P
    WHERE P.Name IN {{LLMQA('First 3 presidents of the U.S?', modifier='{3}')}}
    """
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
