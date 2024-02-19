from typing import Dict, Union, Optional

import pandas as pd

from blendsql.llms._llm import LLM
from blendsql._programs import validate_program
from blendsql.ingredients.ingredient import QAIngredient


class LLMValidate(QAIngredient):
    def run(
        self,
        question: str,
        llm: LLM,
        context: Optional[pd.DataFrame] = None,
        value_limit: Optional[int] = None,
        table_to_title: Optional[Dict[str, str]] = None,
        long_answer: bool = False,
        **kwargs,
    ) -> Union[str, int, float]:
        if context is not None:
            if value_limit is not None:
                context = context.iloc[:value_limit]
        res = llm.predict(
            program=validate_program,
            claim=question,
            serialized_db=context.to_string() if context is not None else "",
            long_answer=long_answer,
            table_title=None,
        )
        return int(res["result"] == "true")
