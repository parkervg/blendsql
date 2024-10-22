from typing import Literal, Tuple
from collections.abc import Collection
from textwrap import dedent

from blendsql.db.utils import double_quote_escape
from blendsql.ingredients.ingredient import AliasIngredient, Ingredient
from blendsql.ingredients.builtin.web_search import BingWebSearch
from blendsql.ingredients.builtin.qa import LLMQA
from blendsql._exceptions import IngredientException


class RAGQA(AliasIngredient):
    def run(
        self, question: str, source: Literal["bing"] = "bing", *args, **kwargs
    ) -> Tuple[str, Collection[Ingredient]]:
        '''Returns a subquery which first fetches relevant context from a source,
        and returns a retrieval-augmented LM generation.

        Arguments:
            question: The query string to use for both retrieval and generation
            source: The source of the retrieved information. Currently only supports 'bing'

        Examples:
            ```python
            from blendsql.ingredients import RAGQA
            # Among the schools with the average score in Math over 560 in the SAT test, how many schools are in the bay area?
            blendsql_query = """
            SELECT COUNT(DISTINCT s.CDSCode)
                FROM schools s
                JOIN satscores sa ON s.CDSCode = sa.cds
                WHERE sa.AvgScrMath > 560
                AND s.County IN {{RAGQA('Which counties are in the Bay Area?')}}
            """
            ingredients = {RAGQA}
            ...
            ```
        '''
        if source == "bing":
            rag_ingredient = BingWebSearch
        else:
            raise IngredientException(
                f"RAGQA not setup to handle source '{source}' yet"
            )
        new_query = dedent(
            f"""
        {{{{
            LLMQA(
                "{double_quote_escape(question)}", 
                (
                    SELECT {{{{
                        {rag_ingredient.__name__}("{double_quote_escape(question)}")
                    }}}} AS "Search Results"
                )
            )
        }}}}
        """
        )
        return (new_query, {LLMQA, rag_ingredient})
