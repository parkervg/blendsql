import pandas as pd
import psutil
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from blendsql import BlendSQL
from blendsql.models import LiteLLM, LlamaCpp


USE_LOCAL_CONSTRAINED_MODEL = True

# Load model, either a local transformers model, or remote provider via LiteLLM
model = None
if True:
    if USE_LOCAL_CONSTRAINED_MODEL:
        model = LlamaCpp(
            model_name_or_path="bartowski/SmolLM2-135M-Instruct-GGUF",
            filename="SmolLM2-135M-Instruct-Q4_K_M.gguf",
            config={
                "n_gpu_layers": -1,
                "n_ctx": 8000,
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
        "Eras": pd.DataFrame(
            {"Years": ["1700-1800", "1800-1900", "1900-2000", "2000-Now"]}
        ),
    },
    model=model,
    verbose=True,
)

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
