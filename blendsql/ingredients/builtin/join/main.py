from typing import List, Optional, Tuple
import re
from colorama import Fore

from blendsql.models import Model, LocalModel
from blendsql._program import Program
from blendsql._logger import logger
from blendsql import _constants as CONST
from blendsql.ingredients.ingredient import JoinIngredient
from blendsql.utils import newline_dedent
from blendsql import generate


class JoinProgram(Program):
    def __call__(
        self,
        model: Model,
        join_criteria: str,
        left_values: List[str],
        right_values: List[str],
        sep: str,
        **kwargs,
    ) -> Tuple[str, str]:
        prompt = ""
        prompt += "You are a database expert in charge of performing a modified `LEFT JOIN` operation. This `LEFT JOIN` is based on a semantic criteria given by the user."
        prompt += f"\nThe left and right value alignment should be separated by '{sep}', with each new `JOIN` alignment goin on a newline. If a given left value has no corresponding right value, give '-' as a response."
        prompt += newline_dedent(
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
        """
        )
        prompt += newline_dedent(
            """
            Criteria: {}

            Left Values:
            {}

            Right Values:
            {}

            Output:
            """.format(
                join_criteria, "\n".join(left_values), "\n".join(right_values)
            )
        )
        # Create this pattern on the fly, and not in infer_gen_constraints
        # since it depends on what our left/right values are
        regex = (
            lambda num_repeats: "(({}){}({})\n)".format(
                "|".join([re.escape(i) for i in left_values]),
                CONST.DEFAULT_ANS_SEP,
                "|".join(
                    [re.escape(i) for i in right_values] + [CONST.DEFAULT_NAN_ANS]
                ),
            )
            + "{"
            + str(num_repeats)
            + "}"
        )
        max_tokens = (
            len(
                model.tokenizer.encode(
                    "".join(left_values)
                    + "".join(right_values)
                    + (CONST.DEFAULT_ANS_SEP * len(left_values)),
                )
            )
            if model.tokenizer is not None
            else None
        )

        if isinstance(model, LocalModel):
            response = generate.regex(
                model,
                prompt=prompt,
                pattern=regex(len(left_values)),
                max_tokens=max_tokens,
                stop_at=["---"],
            )
        else:
            response = generate.text(
                model, prompt=prompt, max_tokens=max_tokens, stop_at=["---"]
            )
        logger.debug(Fore.CYAN + prompt + Fore.RESET)
        logger.debug(Fore.LIGHTCYAN_EX + response + Fore.RESET)
        return (response, prompt)


class LLMJoin(JoinIngredient):
    DESCRIPTION = """
    If we need to do a `join` operation where there is imperfect alignment between table values, use the new function:
        `{{LLMJoin(left_on='table::column', right_on='table::column')}}`
    """

    def run(
        self,
        model: Model,
        left_values: List[str],
        right_values: List[str],
        question: Optional[str] = None,
        **kwargs,
    ) -> dict:
        if question is None:
            question = "Join to same topics."
        result = model.predict(
            program=JoinProgram,
            sep=CONST.DEFAULT_ANS_SEP,
            left_values=left_values,
            right_values=right_values,
            join_criteria=question,
            **kwargs,
        )
        # Post-process language model response
        _result = result.split("\n")
        mapping: dict = {}
        for item in _result:
            if CONST.DEFAULT_ANS_SEP in item:
                k, v = item.rsplit(CONST.DEFAULT_ANS_SEP, 1)
                if any(pred == CONST.DEFAULT_NAN_ANS for pred in {k, v}):
                    continue
                mapping[k] = v
        return mapping
