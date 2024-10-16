from typing import List, Optional, Tuple, Dict, Callable
import json
from colorama import Fore
from pathlib import Path
from attr import attrs, attrib
import guidance

from blendsql.models import Model, LocalModel
from blendsql._program import Program
from blendsql._logger import logger
from blendsql.ingredients.ingredient import JoinIngredient
from blendsql.ingredients.generate import generate, user, assistant
from blendsql.ingredients.utils import initialize_retriever, partialclass

from .examples import AnnotatedJoinExample, JoinExample

DEFAULT_JOIN_FEW_SHOT: List[AnnotatedJoinExample] = [
    AnnotatedJoinExample(**d)
    for d in json.loads(
        open(Path(__file__).resolve().parent / "./default_examples.json", "r").read()
    )
]
MAIN_INSTRUCTION = "You are a database expert in charge of performing a modified `LEFT JOIN` operation. This `LEFT JOIN` is based on a semantic criteria given by the user.f\nIf a given left value has no corresponding right value, give '-' as a response. Stop responding as soon as all left values have been accounted for in the JSON mapping.\n"


class JoinProgram(Program):
    def __call__(
        self,
        model: Model,
        current_example: JoinExample,
        few_shot_examples: List[JoinExample],
        **kwargs,
    ) -> Tuple[Dict[str, str], str]:
        if isinstance(model, LocalModel):
            lm: guidance.models.Model = model.model_obj
            with guidance.user():
                lm += MAIN_INSTRUCTION
            # Add few-shot examples
            for example in few_shot_examples:
                with guidance.user():
                    lm += example.to_string()
                with guidance.assistant():
                    lm += "```json\n" + json.dumps(example.mapping, indent=4) + "\n```"
            with guidance.user():
                lm += current_example.to_string()
            prompt = lm._current_prompt()

            @guidance(stateless=True, dedent=False)
            def make_predictions(lm, left_values, right_values):
                lm += "```json\n{"
                gen_f = guidance.select(options=right_values)
                for idx, value in enumerate(left_values):
                    lm += (
                        f'\n\t"{value}": '
                        + guidance.capture(gen_f, name=value)
                        + ("," if idx + 1 != len(right_values) else "")
                    )
                lm += "```"
                return lm

            with guidance.assistant():
                lm += make_predictions(
                    left_values=current_example.left_values,
                    right_values=current_example.right_values,
                )
            return (lm._variables, prompt)
        else:
            # Use 'old' style prompt for remote models
            messages = []
            messages.append(user(MAIN_INSTRUCTION))
            # Add few-shot examples
            for example in few_shot_examples:
                messages.append(user(example.to_string()))
                messages.append(
                    assistant(
                        "```json\n" + json.dumps(example.mapping, indent=4) + "\n```"
                    )
                )
            messages.append(user(current_example.to_string()))
            prompt = "".join([i["content"] for i in messages])
            response = (
                generate(model, messages=messages)
                .removeprefix("```json")
                .removesuffix("```")
            )
            # Post-process language model response
            try:
                mapping: dict = json.loads(response)
            except json.decoder.JSONDecodeError:
                mapping = {}
                logger.debug(
                    Fore.RED
                    + f"LLMJoin failed to return valid JSON!\nGot back '{response}'"
                )
        return (mapping, prompt)


@attrs
class LLMJoin(JoinIngredient):
    DESCRIPTION = """
    If we need to do a `join` operation where there is imperfect alignment between table values, use the new function:
        `{{LLMJoin(left_on='table::column', right_on='table::column')}}`
    """
    model: Model = attrib(default=None)
    few_shot_retriever: Callable[[str], List[AnnotatedJoinExample]] = attrib(
        default=None
    )

    @classmethod
    def from_args(
        cls,
        model: Model = None,
        use_skrub_joiner: bool = True,
        few_shot_examples: List[dict] = None,
        k: Optional[int] = None,
    ):
        """Creates a partial class with predefined arguments.

        Args:
            few_shot_examples: A list of AnnotatedJoinExamples dictionaries for few-shot learning.
                If not specified, will use [default_examples.json](https://github.com/parkervg/blendsql/blob/main/blendsql/ingredients/builtin/join/default_examples.json) as default.
            use_skrub_joiner: Whether to use the skrub joiner. Defaults to True.
            k: Determines number of few-shot examples to use for each ingredient call.
                Default is None, which will use all few-shot examples on all calls.
                If specified, will initialize a haystack-based DPR retriever to filter examples.

        Returns:
            Type[JoinIngredient]: A partial class of JoinIngredient with predefined arguments.

        Examples:
            ```python
            from blendsql import blend, LLMJoin
            from blendsql.ingredients.builtin import DEFAULT_JOIN_FEW_SHOT

            ingredients = {
                LLMJoin.from_args(
                    few_shot_examples=[
                        *DEFAULT_JOIN_FEW_SHOT,
                        {
                            "join_criteria": "Join the state to its capital.",
                            "left_values": ["California", "Massachusetts", "North Carolina"],
                            "right_values": ["Sacramento", "Boston", "Chicago"],
                            "mapping": {
                                "California": "Sacramento",
                                "Massachusetts": "Boston",
                                "North Carolina": "-"
                            }
                        }
                    ],
                    # Will fetch `k` most relevant few-shot examples using embedding-based retriever
                    k=2
                )
            }
            smoothie = blend(
                query=blendsql,
                db=db,
                ingredients=ingredients,
                default_model=model,
            )
            ```
        """
        if few_shot_examples is None:
            few_shot_examples = DEFAULT_JOIN_FEW_SHOT
        else:
            few_shot_examples = [
                AnnotatedJoinExample(**d) if isinstance(d, dict) else d
                for d in few_shot_examples
            ]
        few_shot_retriever = initialize_retriever(examples=few_shot_examples, k=k)
        return partialclass(
            cls,
            model=model,
            few_shot_retriever=few_shot_retriever,
            use_skrub_joiner=use_skrub_joiner,
        )

    def run(
        self,
        model: Model,
        left_values: List[str],
        right_values: List[str],
        question: Optional[str] = None,
        few_shot_retriever: Callable[[str], List[AnnotatedJoinExample]] = None,
        **kwargs,
    ) -> dict:
        if question is None:
            question = "Join to same topics."
        if few_shot_retriever is None:
            few_shot_retriever = lambda *_: DEFAULT_JOIN_FEW_SHOT
        current_example = JoinExample(
            **{
                "join_criteria": question,
                "left_values": left_values,
                "right_values": right_values,
            }
        )
        few_shot_examples: List[AnnotatedJoinExample] = few_shot_retriever(
            current_example.to_string()
        )
        mapping = model.predict(
            program=JoinProgram,
            current_example=JoinExample(
                **{
                    "join_criteria": question,
                    "left_values": left_values,
                    "right_values": right_values,
                }
            ),
            few_shot_examples=few_shot_examples,
            **kwargs,
        )
        return {k: v for k, v in mapping.items() if v != "-"}
