from pathlib import Path

from blendsql.models import OpenaiLLM
from blendsql.ingredients import LLMQA, LLMMap
from blendsql.db import SQLite
from blendsql import blend


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
    ingredients = {LLMQA, LLMMap}
    db = load_tag_database("california_schools")
    blendsql_query = """
    SELECT s.Phone 
        FROM satscores ss 
        JOIN schools s ON ss.cds = s.CDSCode 
        WHERE {{LLMMap('Is this county in Southern California?', 's::County')}} = TRUE
        ORDER BY ss.AvgScrRead ASC 
        LIMIT 1
    """
    smoothie = blend(
        query=blendsql_query,
        default_model=OpenaiLLM("gpt-4o-mini"),
        ingredients=ingredients,
        db=db,
        verbose=True,
    )
    print(smoothie.df)
