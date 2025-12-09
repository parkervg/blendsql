from blendsql.ingredients import LLMMap
import pandas as pd
from blendsql import BlendSQL
from blendsql.search import ColbertWikipediaSearch
from blendsql.models import TransformersLLM

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
        model=TransformersLLM("HuggingFaceTB/SmolLM2-135M-Instruct"),
        verbose=True,
    )
    _ = bsql.model.model_obj

    WikipediaSearchMap = LLMMap.from_args(
        context_searcher=ColbertWikipediaSearch(
            k=1,  # Retrieve 1 document for each scalar value on the map call
        ),
    )
    bsql.ingredients = {
        WikipediaSearchMap,
    }

    # What is the name of the oldest person?
    # The `WikipediaSearchMap` will aggregate context for each entry using the ColbertWikipediaSearch,
    #   passing it as context to yield an integer
    smoothie = bsql.execute(
        """
        SELECT Name FROM world_aquatic_championships w
        WHERE event NOT LIKE '%team%'
        ORDER BY {{WikipediaSearchMap('What year was {} born?', w.Name)}} ASC LIMIT 1
        """
    )

    print(smoothie.df)
    # ┌─────────────┐
    # │ Name        │
    # ├─────────────┤
    # │ Ryan Lochte │
    # └─────────────┘
    from blendsql.configure import GLOBAL_HISTORY

    print(GLOBAL_HISTORY)
