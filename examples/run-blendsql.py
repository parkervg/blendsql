import pandas as pd

from blendsql import BlendSQL
from blendsql.ingredients import LLMMap, LLMQA, LLMJoin
from blendsql.models import TransformersLLM, LiteLLM, LlamaCpp
from pip._vendor.certifi import where

USE_LOCAL_CONSTRAINED_MODEL = True

# Load model, either a local transformers model, or remote provider via LiteLLM
if USE_LOCAL_CONSTRAINED_MODEL:
    # model = TransformersLLM(
    #     "meta-llama/Llama-3.2-3B-Instruct", config={"device_map": "auto"}
    # )  # Local models enable BlendSQL's predicate-guided constrained decoding

    model = LlamaCpp(
        "Meta-Llama-3-8B-Instruct.Q6_K.gguf",
        "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        config={"n_gpu_layers": -1},
        caching=False
    )
else:
    model = LiteLLM("openai/gpt-4o-mini")

# Prepare our BlendSQL connection
bsql = BlendSQL(
    {
        "People": pd.DataFrame(
            {
                "Name": [
                    "George Washington",
                    "John Adams",
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
    # ingredients={LLMMap, LLMQA, LLMJoin},
    ingredients={LLMMap.from_args(k=0), LLMQA.from_args(k=0), LLMJoin.from_args(k=0)},
    model=model,
    verbose=True,
)

smoothie = bsql.execute(
    """
    WITH musicians AS (
        SELECT * FROM People p WHERE
        {{LLMMap('Is a singer?', 'p::Name')}} = True
    ) SELECT * FROM musicians WHERE
    musicians.Name = {{LLMQA('Who wrote the song espresso?')}}
    """,
    infer_gen_constraints=True,
)

print(smoothie.df)
# ┌───────────────────┬───────────────────────────────────────────────────────┐
# │ Name              │ Known_For                                             │
# ├───────────────────┼───────────────────────────────────────────────────────┤
# │ George Washington │ Established federal government, First U.S. Preside... │
# │ John Adams │ XYZ Affair, Alien and Sedition Acts                   │
# │ Thomas Jefferson  │ Louisiana Purchase, Declaration of Independence       │
# └───────────────────┴───────────────────────────────────────────────────────┘
print(smoothie.summary())
# ┌────────────┬──────────────────────┬─────────────────┬─────────────────────┐
# │   Time (s) │   # Generation Calls │   Prompt Tokens │   Completion Tokens │
# ├────────────┼──────────────────────┼─────────────────┼─────────────────────┤
# │    1.25158 │                    1 │             296 │                  16 │
# └────────────┴──────────────────────┴─────────────────┴─────────────────────┘
