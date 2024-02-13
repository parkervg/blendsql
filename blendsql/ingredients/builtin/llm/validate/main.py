from typing import Dict, Union, Optional

import pandas as pd

from blendsql.ingredients.builtin.llm.utils import (
    construct_gen_clause,
)
from blendsql.ingredients.builtin.llm.llm import LLM
from blendsql import _programs as programs
from blendsql import _constants as CONST
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
        gen_clause: str = construct_gen_clause(
            gen_type="select",
            options=["true", "false"],
            **llm.gen_kwargs,
        )
        program: str = (
            programs.VALIDATE_PROGRAM_CHAT(gen_clause)
            if llm.model_name_or_path in CONST.OPENAI_CHAT_LLM
            else programs.VALIDATE_PROGRAM_COMPLETION(gen_clause)
        )
        res = llm.predict(
            program=program,
            question=question,
            serialized_db=context.to_string() if context is not None else "",
            long_answer=long_answer,
            table_title=None,
        )
        return int(res["result"] == "true")
