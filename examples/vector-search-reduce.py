import pandas as pd
import torch.cuda

from blendsql import BlendSQL
from blendsql.ingredients import LLMQA
from blendsql.models import LlamaCpp, LiteLLM
from blendsql.search import FaissVectorStore

bsql = BlendSQL(
    {
        "schools": pd.DataFrame(
            [
                {
                    "County": "Orange",
                    "StatusType": "Active",
                    "School": "Woodbury Elementary",
                },
                {
                    "County": "Kings",
                    "StatusType": "Merged",
                    "School": "Washington Elementary",
                },
                {
                    "County": "Alameda",
                    "StatusType": "Merged",
                    "School": "Whiteford (June) School",
                },
                {
                    "County": "Santa Clara",
                    "StatusType": "Active",
                    "School": "Bullis Charter",
                },
                {
                    "County": "Riverside",
                    "StatusType": "Merged",
                    "School": "Sunnymead Middle",
                },
                {
                    "County": "San Diego",
                    "StatusType": "Closed",
                    "School": "Mt. Laguna Elementary",
                },
                {
                    "County": "Fresno",
                    "StatusType": "Closed",
                    "School": "KIPP Academy Fresno",
                },
                {"County": "Tulare", "StatusType": "Active", "School": None},
                {
                    "County": "Trinity",
                    "StatusType": "Active",
                    "School": "Burnt Ranch Elementary",
                },
                {
                    "County": "Tuolumne",
                    "StatusType": "Closed",
                    "School": "Tuolumne County Community Day Middle",
                },
            ]
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
        "european_countries": pd.DataFrame(
            [
                {
                    "Country": "Portugal",
                    "Area (km²)": 91568,
                    "Population (As of 2011)": 10555853,
                    "Population density (per km²)": 115.2,
                    "Capital": "Lisbon",
                },
                {
                    "Country": "Sweden",
                    "Area (km²)": 449964,
                    "Population (As of 2011)": 9088728,
                    "Population density (per km²)": 20.1,
                    "Capital": "Stockholm",
                },
                {
                    "Country": "United Kingdom",
                    "Area (km²)": 244820,
                    "Population (As of 2011)": 62300000,
                    "Population density (per km²)": 254.4,
                    "Capital": "London",
                },
            ]
        ),
    },
    model=LlamaCpp(
        "Meta-Llama-3.1-8B-Instruct.Q6_K.gguf",
        "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF",
        config={"n_gpu_layers": -1, "n_ctx": 9600},
    )
    if torch.cuda.is_available()
    else LiteLLM("openai/gpt-4o"),
    verbose=True,
)

# Create embeddings for the concatenation of `title` and `content` columns
DocumentSearch = LLMQA.from_args(
    searcher=FaissVectorStore(
        documents=bsql.db.execute_to_list(
            "SELECT CONCAT(title, ' | ', content) FROM documents;"
        ),
        k=1,  # Retrieve 1 document on each search
    ),
)

# Replacement scan allows us to reference new `DocumentSearch` in query
bsql.ingredients = {DocumentSearch}

smoothie = bsql.execute(
    """
SELECT Capital FROM european_countries
WHERE Country = {{
    DocumentSearch('Which country is E.F. Codd from?')
}}
"""
)
print(smoothie.df)
# ┌───────────┐
# │ Capital   │
# ├───────────┤
# │ London    │
# └───────────┘


#  The below query will use our faiss vector store to fetch relevant context and respond to question
#  There's a school that is currently active, in the Bay Area. Who played basketball in the county that the school is in?
smoothie = bsql.execute(
    """
WITH bay_area_county AS (
    SELECT County FROM schools s
    WHERE {{LLMMap('Is this county in the California Bay Area?', s.County)}}
    AND s.StatusType = 'Active' LIMIT 1
) SELECT {{DocumentSearch('Who played basketball at a school in {}?', bay_area_count.County)}} AS answer
"""
)
print(smoothie.df)
# ┌────────────┐
# │ answer     │
# ├────────────┤
# │ Steve Nash │
# └────────────┘
