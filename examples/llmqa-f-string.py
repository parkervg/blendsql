import pandas as pd
import torch.cuda

from blendsql import BlendSQL
from blendsql.ingredients import LLMQA
from blendsql.models import LlamaCpp, LiteLLM
from blendsql.search import FaissVectorStore

bsql = BlendSQL(
    {
        "schools": pd.DataFrame(
            {"Status": "open", "County": "Santa Clara"},
            {"Status": "closed", "County": "Alameda"},
        ),
        "documents": pd.DataFrame(
            [
                {
                    "title": "Steve Nash",
                    "content": "Steve Nash played college basketball at Santa Clara University",
                },
                {
                    "title": "E.F. Codd",
                    "content": 'Edgar Frank "Ted" Codd (19 August 1923 – 18 April 2003) was a British computer scientist who, while working for IBM, invented the relational model for database management, the theoretical basis for relational databases and relational database management systems.',
                },
                {
                    "title": "George Washington (February 22, 1732 – December 14, 1799) was a Founding Father and the first president of the United States, serving from 1789 to 1797."
                },
                {
                    "title": "Thomas Jefferson",
                    "content": "Thomas Jefferson (April 13, 1743 – July 4, 1826) was an American Founding Father and the third president of the United States from 1801 to 1809.",
                },
                {
                    "title": "John Adams",
                    "content": "John Adams (October 30, 1735 – July 4, 1826) was an American Founding Father who was the second president of the United States from 1797 to 1801.",
                },
            ]
        ),
    },
    model=LlamaCpp(
        "Meta-Llama-3.1-8B-Instruct.Q6_K.gguf",
        "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF",
        config={"n_gpu_layers": -1},
    )
    if torch.cuda.is_available()
    else LiteLLM("openai/gpt-4o"),
    verbose=True,
)

# Create embeddings for the concatenation of `title` and `content` columns
DocumentSearch = LLMQA.from_args(
    vector_store=FaissVectorStore(
        documents=bsql.db.execute_to_list(
            "SELECT CONCAT(title, ' | ', content) FROM documents;"
        ),
        k=1,  # Retrieve 1 document on each search
    ),
)
# Replacement scan allows us to reference new `DocumentSearch` in query
bsql.ingredients = {DocumentSearch}

# Who played basketball in the County of the school that's open?
# The `DocumentSearch` ingredient will swap in the value from the
#   `bay_area_county` CTE, which will be 'Santa Clara'.
# Then, we search over the Faiss document index to fetch relevant context.
smoothie = bsql.execute(
    """
WITH bay_area_county AS (
    SELECT County FROM schools s
    WHERE s.Status = 'open' LIMIT 1
) SELECT {{DocumentSearch('Who played basketball at a school in {}?', bay_area_county.County)}} AS answer
"""
)
print(smoothie.df)
# ┌────────────┐
# │ answer     │
# ├────────────┤
# │ Steve Nash │
# └────────────┘
