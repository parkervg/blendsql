from blendsql.ingredients import LLMMap
import pandas as pd
from blendsql import BlendSQL
from blendsql.search import FaissVectorStore
from blendsql.models import LlamaCpp, LiteLLM
import torch


if __name__ == "__main__":
    bsql = BlendSQL(
        {
            "world_aquatic_championships": pd.DataFrame(
                [
                    {
                        "Medal": "Gold",
                        "Name": "Andrew Gemmell, Sean Ryan, Ashley Twichell",
                        "Sport": "Open water swimming",
                        "Event": "5 km team event",
                        "Time/Score": "57:00.6",
                        "Date": "July 21",
                    },
                    {
                        "Medal": "Gold",
                        "Name": "Dana Vollmer",
                        "Sport": "Swimming",
                        "Event": "Women's 100 m butterfly",
                        "Time/Score": "56.87",
                        "Date": "July 25",
                    },
                    {
                        "Medal": "Gold",
                        "Name": "Ryan Lochte",
                        "Sport": "Swimming",
                        "Event": "Men's 200 m freestyle",
                        "Time/Score": "1:44.44",
                    },
                    {
                        "Medal": "Gold",
                        "Name": "Rebecca Soni",
                        "Sport": "Swimming",
                        "Event": "Women's 100 m breaststroke",
                        "Time/Score": "1:05.05",
                        "Date": "July 26",
                    },
                    {
                        "Medal": "Gold",
                        "Name": "Elizabeth Beisel",
                        "Sport": "Swimming",
                        "Event": "Women's 400 m individual medley",
                        "Time/Score": "4:31.78",
                        "Date": "July 31",
                    },
                    {
                        "Medal": "Gold",
                        "Name": "Ryan Lochte",
                        "Sport": "Swimming",
                        "Event": "Men's 400 m individual medley",
                        "Time/Score": "4:07.13",
                        "Date": "July 28",
                    },
                ]
            ),
            "documents": pd.DataFrame(
                [
                    {
                        "title": "Ryan Lochte",
                        "content": "Ryan Steven Lochte (/ˈlɒkti/ LOK-tee; born August 3, 1984) is an American former[2] competition swimmer and 12-time Olympic medalist.",
                    },
                    {
                        "title": "Elizabeth Beisel",
                        "content": "Elizabeth Lyon Beisel (/ˈbaɪzəl/; born August 18, 1992) is an American competition swimmer who specializes in backstroke and individual medley events.",
                    },
                    {
                        "title": "Rebecca Soni",
                        "content": "Rebecca Soni (born March 18, 1987) is an American former competition swimmer and breaststroke specialist who is a six-time Olympic medalist.",
                    },
                    {
                        "title": "Dana Vollmer",
                        "content": "Dana Whitney Vollmer (born November 13, 1987) is a former American competition swimmer, five-time Olympic gold medalist, and former world record-holder.",
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
    _ = bsql.model.model_obj

    WikipediaSearchMap = LLMMap.from_args(
        vector_store=FaissVectorStore(
            documents=bsql.db.execute_to_list("SELECT content FROM documents;"),
            k=1,  # Retrieve 1 document for each scalar value on the map call
        ),
    )
    bsql.ingredients = {
        WikipediaSearchMap,
    }

    # What is the name of the oldest person whose result, not including team race, was above 2 minutes?
    # The `WikipediaSearchMap` will aggregate context for each entry using the FaissVectorStore,
    #   passing it as context to yield an integer
    smoothie = bsql.execute(
        """
        WITH t AS (
            SELECT Name FROM "world_aquatic_championships" wc
            WHERE {{LLMMap('Is this a team event?', 'wc::Event')}} = FALSE
            /* Use DuckDB functions to parse minutes */
            AND TRY_CAST(REGEXP_EXTRACT("time/score", '^([0-9]+):', 1) AS INT) >= 2
        ) SELECT Name FROM t
        ORDER BY {{LLMMap('What year were they born?', values='t::Name')}} ASC LIMIT 1
        """
    )

    print(smoothie.df)
    # ┌─────────────┐
    # │ Name        │
    # ├─────────────┤
    # │ Ryan Lochte │
    # └─────────────┘
