from typing import List
from guidance import gen
from textwrap import dedent

from blendsql.models._model import Model
from blendsql._program import Program
from blendsql import _constants as CONST
from blendsql.ingredients.ingredient import JoinIngredient


class JoinProgram(Program):
    def __call__(
        self,
        join_criteria: str,
        left_values: List[str],
        right_values: List[str],
        sep: str,
        **kwargs,
    ):
        left_values = "\n".join(left_values)
        right_values = "\n".join(right_values)
        with self.systemcontext:
            self.model += "You are a database expert in charge of performing a modified `LEFT JOIN` operation. This `LEFT JOIN` is based on a semantic criteria given by the user."
            self.model += f"\nThe left and right value alignment should be separated by '{sep}', with each new `JOIN` alignment goin on a newline. If a given left value has no corresponding right value, give '-' as a response."
        with self.usercontext:
            if self.few_shot:
                self.model += dedent(
                    """
                Criteria: Join to same topics.

                Left Values: 
                joshua fields
                bob brown
                ron ryan

                Right Values: 
                ron ryan
                colby mules
                bob brown (ice hockey)
                josh fields (pitcher)

                Output: 
                joshua fields;josh fields (pitcher)
                bob brown;bob brown (ice hockey)
                ron ryan;ron ryan

                --- 

                Criteria: Align the fruit to their corresponding colors.

                Left Values: 
                apple
                banana
                blueberry
                orange

                Right Values: 
                blue
                yellow
                red

                Output: 
                apple;red
                banana;yellow
                blueberry;blue
                orange;-

                ---
                """
                )
            self.model += dedent(
                f"""
                Criteria: {join_criteria}

                Left Values: {left_values}

                Right Values: {right_values}

                Output:
                """
            )
        with self.assistantcontext:
            self.model += gen(name="result", **self.gen_kwargs)
        return self.model


class LLMJoin(JoinIngredient):
    def run(
        self,
        left_values: List[str],
        right_values: List[str],
        model: Model,
        join_criteria: str = "Join to same topics.",
        **kwargs,
    ) -> dict:
        res = model.predict(
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
