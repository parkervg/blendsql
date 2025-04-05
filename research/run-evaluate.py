from pathlib import Path

from blendsql import BlendSQL
from blendsql.models import LiteLLM
from blendsql.ingredients import LLMQA, LLMMap, BingWebSearch, RAGQA
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
    ingredients = {LLMQA, LLMMap, BingWebSearch, RAGQA}
    bsql = BlendSQL(
        load_tag_database("california_schools"),
        ingredients={LLMQA, LLMMap, BingWebSearch, RAGQA},
        model=LiteLLM("openai/gpt-4o"),
    )
    # Among the schools with the average score in Math over 560 in the SAT test, how many schools are in the bay area?
    smoothie = bsql.execute(
        """
        SELECT COUNT(DISTINCT s.CDSCode) 
            FROM schools s 
            JOIN satscores sa ON s.CDSCode = sa.cds 
            WHERE sa.AvgScrMath > 560 
            AND s.County IN {{RAGQA('Which counties are in the Bay Area?')}}
        """
    )
    print(smoothie.df)
    print(smoothie.meta.process_time_seconds)
    # blendsql_query = """
    # SELECT s.Phone
    #     FROM satscores ss
    #     JOIN schools s ON ss.cds = s.CDSCode
    #     WHERE county IN {{
    #         LLMQA(
    #             'Which counties are in the Bay Area?',
    #             (
    #                 SELECT {{
    #                     BingWebSearch('Counties in the Bay Area')
    #                 }} AS "Search Results"
    #             )
    #         )
    #     }}
    #     ORDER BY ss.AvgScrRead ASC
    #     LIMIT 1
    # """
