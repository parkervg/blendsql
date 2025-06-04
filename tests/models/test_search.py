import pytest
import pandas as pd

from blendsql import BlendSQL
from blendsql.ingredients import LLMMap
from blendsql.search import FaissVectorStore


@pytest.fixture(scope="session")
def bsql() -> BlendSQL:
    return BlendSQL(
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
    )


def test_faiss_search(bsql, model):
    WikipediaSearchMap = LLMMap.from_args(
        searcher=FaissVectorStore(
            documents=bsql.db.execute_to_list(
                "SELECT CONCAT(title, ' | ', content) FROM documents;"
            ),
            k=1,
        ),
    )
    bsql.ingredients = {
        WikipediaSearchMap,
    }
    _ = bsql.execute(
        """
         WITH t AS (
            SELECT Name FROM "world_aquatic_championships"
            WHERE {{LLMMap('Is this a team event?', Event)}} = FALSE
            AND {{LLMMap('Is this time over 2 minutes?', "Time/Score")}} = TRUE
        ) SELECT Name FROM t
        ORDER BY {{LLMMap('What year was {} born?', t.Name)}} ASC LIMIT 1
        """,
        model=model,
    )
