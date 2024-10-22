from typing import Literal
from textwrap import dedent
from attr import attrs

from blendsql.db.utils import double_quote_escape
from blendsql.ingredients.ingredient import AliasIngredient
from blendsql.ingredients.builtin.qa import LLMQA
from blendsql.ingredients.builtin.web_search import BingWebSearch
from blendsql._exceptions import IngredientException


@attrs
class RAGQA(AliasIngredient):
    def run(
        self, question: str, source: Literal["bing"] = "bing", *args, **kwargs
    ) -> str:
        if source == "bing":
            rag_ingredient = BingWebSearch
        else:
            raise IngredientException(
                f"RAGQA not setup to handle source '{source}' yet"
            )
        return (
            dedent(
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
            ),
            {LLMQA, rag_ingredient},
        )
