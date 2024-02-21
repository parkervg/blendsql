from typing import Dict, Union, Optional, List

import pandas as pd
from blendsql.llms._llm import LLM
from blendsql._programs import QAProgram
from blendsql.ingredients.ingredient import QAIngredient
from blendsql.db.utils import single_quote_escape


class LLMQA(QAIngredient):
    def run(
        self,
        question: str,
        llm: LLM,
        options: Optional[List[str]] = None,
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
            program=QAProgram,
            options=options,
            question=question,
            serialized_db=context.to_string() if context is not None else "",
            long_answer=long_answer,
            table_title=None,
            **kwargs,
        )
        return "'{}'".format(single_quote_escape(res["result"].strip()))
