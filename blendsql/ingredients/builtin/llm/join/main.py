from typing import List
from blendsql.ingredients.builtin.llm.utils import (
    construct_gen_clause,
)

from blendsql.ingredients.builtin.llm.endpoint import Endpoint
from blendsql import _programs as programs
from blendsql import _constants as CONST
from blendsql.ingredients.ingredient import JoinIngredient


class LLMJoin(JoinIngredient):
    def run(
        self,
        left_values: List[str],
        right_values: List[str],
        endpoint: Endpoint,
        join_criteria: str = "Join to same topics.",
        **kwargs,
    ) -> dict:
        gen_clause: str = construct_gen_clause(
            **endpoint.gen_kwargs,
        )
        program: str = (
            programs.JOIN_PROGRAM_CHAT(gen_clause)
            if endpoint.endpoint_name in CONST.OPENAI_CHAT_LLM
            else programs.JOIN_PROGRAM_COMPLETION(gen_clause)
        )

        res = endpoint.predict(
            program=program,
            sep=CONST.DEFAULT_ANS_SEP,
            left_values="\n".join(left_values),
            right_values="\n".join(right_values),
            join_criteria=join_criteria,
        )

        _result = res["result"].split("\n")
        result: dict = {}
        for item in _result:
            if CONST.DEFAULT_ANS_SEP in item:
                k, v = item.rsplit(CONST.DEFAULT_ANS_SEP, 1)
                result[k] = v
        return result
