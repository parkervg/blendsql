from typing import List

from blendsql.llms._llm import LLM
from blendsql._programs import JoinProgram
from blendsql import _constants as CONST
from blendsql.ingredients.ingredient import JoinIngredient


class LLMJoin(JoinIngredient):
    def run(
        self,
        left_values: List[str],
        right_values: List[str],
        llm: LLM,
        join_criteria: str = "Join to same topics.",
        **kwargs,
    ) -> dict:
        res = llm.predict(
            program=JoinProgram,
            sep=CONST.DEFAULT_ANS_SEP,
            left_values=left_values,
            right_values=right_values,
            join_criteria=join_criteria,
            **kwargs,
        )

        _result = res["result"].split("\n")
        result: dict = {}
        for item in _result:
            if CONST.DEFAULT_ANS_SEP in item:
                k, v = item.rsplit(CONST.DEFAULT_ANS_SEP, 1)
                if any(pred == CONST.DEFAULT_NAN_ANS for pred in {k, v}):
                    continue
                result[k] = v
        return result
