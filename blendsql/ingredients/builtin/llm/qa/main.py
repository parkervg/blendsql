from typing import Dict, Union, Optional, List
import re

import pandas as pd

from blendsql.ingredients.builtin.llm.utils import (
    construct_gen_clause,
)
from blendsql.ingredients.builtin.llm.endpoint import Endpoint
from blendsql import _programs as programs
from blendsql import _constants as CONST
from blendsql.ingredients.ingredient import QAIngredient
from blendsql.db.utils import single_quote_escape


class LLMQA(QAIngredient):
    def run(
        self,
        question: str,
        endpoint: Endpoint,
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
        # Ensure options don't have a common prefix
        # This will be removed later
        # Related to this issue: https://github.com/guidance-ai/guidance/issues/232#issuecomment-1597125256
        if options is not None:
            options = [f"{idx}): {item}" for idx, item in enumerate(options)]
        gen_clause: str = construct_gen_clause(
            gen_type="select" if options else "gen",
            options=options,
            **endpoint.gen_kwargs,
        )
        program: str = (
            programs.QA_PROGRAM_CHAT(gen_clause)
            if endpoint.endpoint_name in CONST.OPENAI_CHAT_LLM
            else programs.QA_PROGRAM_COMPLETION(gen_clause)
        )
        res = endpoint.predict(
            program=program,
            options_dict=[{"option": item} for item in options] if options else None,
            question=question,
            serialized_db=context.to_string() if context is not None else "",
            long_answer=long_answer,
            table_title=None,
        )
        if options is None:
            _result = res["result"]
        else:
            _result = re.sub(r"^\d+\):", "", res["result"])
        return "'{}'".format(single_quote_escape(_result.strip()))
