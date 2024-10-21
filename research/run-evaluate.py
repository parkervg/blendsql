from pathlib import Path

import blendsql
from blendsql.models import OpenaiLLM
from blendsql.ingredients import LLMQA, LLMMap, BingWebSearch
from blendsql.db import SQLite


def load_tag_database(name: str) -> SQLite:
    return SQLite(
        Path("./research/data/bird-sql/dev_20240627/dev_databases/")
        / name
        / f"{name}.sqlite"
    )


if __name__ == "__main__":
    # df = pd.read_csv("./research/data/tag-benchmark/tag_queries.csv")
    # for _, row in df.iterrows():
    #     db = load_tag_database(row['DB used'])
    #     print()
    """
    7.95 seconds with 1
    4.7155 with 5
    """
    ingredients = {LLMQA, LLMMap, BingWebSearch}
    db = load_tag_database("california_schools")
    blendsql_query = """
    SELECT s.Phone 
        FROM satscores ss 
        JOIN schools s ON ss.cds = s.CDSCode 
        WHERE county IN {{
            LLMQA(
                'Which counties are in the Bay Area?', 
                (
                    {{
                        BingWebSearch('Counties in the Bay Area')
                    }}
                )
            )
        }}
        ORDER BY ss.AvgScrRead ASC 
        LIMIT 1
    """
    smoothie = blendsql.blend(
        query=blendsql_query,
        default_model=OpenaiLLM("gpt-4o-mini", caching=False),
        ingredients=ingredients,
        db=db,
        verbose=True,
    )
    print(smoothie.df)
    print(smoothie.meta.process_time_seconds)
