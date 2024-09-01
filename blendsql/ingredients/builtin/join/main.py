from typing import List, Optional, Tuple
import guidance

from blendsql.models import Model, LocalModel
from blendsql._program import Program
from blendsql import _constants as CONST
from blendsql.ingredients.ingredient import JoinIngredient
from blendsql.utils import newline_dedent
from blendsql.ingredients.generate import generate


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
        if isinstance(model, LocalModel):
            m: guidance.models.Model = model.model_obj
            with guidance.system():
                m += "You are a database expert in charge of performing a modified `LEFT JOIN` operation. This `LEFT JOIN` is based on a semantic criteria given by the user."
                m += f"\nThe left and right value alignment should be separated by '{sep}', with each new `JOIN` alignment goin on a newline. If a given left value has no corresponding right value, give '-' as a response."
                m += newline_dedent(
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
                {
                    "joshua fields": "josh fields (pitcher)",
                    "bob brown": "bob brown (ice hockey)",
                    "ron ryan": "ron ryan"
                }
        
                ---
                """
                )
            with guidance.user():
                m += newline_dedent(
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
            prompt = m._current_prompt()

            @guidance(stateless=True, dedent=False)
            def make_predictions(lm, left_values, right_values):
                lm += "{"
                gen_f = guidance.select(options=right_values)
                for idx, value in enumerate(left_values):
                    lm += (
                        f'\n\t"{value}": '
                        + guidance.capture(gen_f, name=value)
                        + ("," if idx + 1 != len(right_values) else "")
                    )
                return lm

            with guidance.assistant():
                m += make_predictions(
                    left_values=left_values, right_values=right_values
                )
            return (m._variables, prompt)
        else:
            # Use 'old' style prompt for remote models
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
            response = generate(
                model, prompt=prompt, max_tokens=max_tokens, stop_at=["---"]
            )
            # Post-process language model response
            _result = response.split("\n")
            mapping: dict = {}
            for item in _result:
                if CONST.DEFAULT_ANS_SEP in item:
                    k, v = item.rsplit(CONST.DEFAULT_ANS_SEP, 1)
                    if any(pred == CONST.DEFAULT_NAN_ANS for pred in {k, v}):
                        continue
                    mapping[k] = v
            return (mapping, prompt)


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
        mapping = model.predict(
            program=JoinProgram,
            sep=CONST.DEFAULT_ANS_SEP,
            left_values=left_values,
            right_values=right_values,
            join_criteria=question,
            **kwargs,
        )
        return {k: v for k, v in mapping.items() if v != CONST.DEFAULT_NAN_ANS}
