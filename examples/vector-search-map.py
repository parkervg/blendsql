import pandas as pd
from blendsql import BlendSQL
from blendsql.models import LlamaCpp

import psutil

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
            )
        },
        model=LlamaCpp(
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
        ),
        verbose=True,
    )
    _ = bsql.model.model_obj

    from blendsql.search import FaissVectorStore
    from blendsql.ingredients import LLMMap

    # We can also define a local FAISS vector store
    context_searcher = FaissVectorStore(
        model_name_or_path="sentence-transformers/all-mpnet-base-v2",
        documents=[
            "Ryan Steven Lochte (/ˈlɒkti/ LOK-tee; born August 3, 1984) is an American former[2] competition swimmer and 12-time Olympic medalist.",
            "Rebecca Soni (born March 18, 1987) is an American former competition swimmer and breaststroke specialist.",
            "Elizabeth Lyon Beisel (/ˈbaɪzəl/; born August 18, 1992) is an American competition swimmer who specializes in backstroke and individual medley events.",
        ],
        k=1,
    )

    DocumentSearchMap = LLMMap.from_args(context_searcher=context_searcher)

    # This line registers our new function in our `BlendSQL` connection context
    # Replacement scans allow us to now reference the function by the variable name we initialized it to (`DocumentSearchMap`)
    bsql.ingredients = {DocumentSearchMap}

    # Define a blendsql program to answer: 'What is the name of the oldest person who won gold?'
    smoothie = bsql.execute(
        """
       WITH t AS (
            SELECT Name FROM "world_aquatic_championships"
            WHERE {{LLMMap('Is this a team event?', Event)}} = FALSE
            AND {{LLMMap('Is this time over 2 minutes?', "Time/Score")}} = TRUE
        ) SELECT Name FROM t
        ORDER BY {{LLMMap('What year was {} born?', t.Name)}} ASC LIMIT 1
        """
    )
    smoothie.print_summary()
