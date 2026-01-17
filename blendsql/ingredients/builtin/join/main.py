import os
from typing import Callable
import json
from pathlib import Path
from dataclasses import dataclass, field
from textwrap import dedent

from blendsql.configure import add_to_global_history
from blendsql.models import Model, ConstrainedModel
from blendsql.models.utils import user, assistant
from blendsql.models.constrained.utils import LMString, maybe_load_lm
from blendsql.common.logger import logger, Color
from blendsql.ingredients.ingredient import JoinIngredient, LMFunctionException
from blendsql.ingredients.utils import initialize_retriever, partialclass
from blendsql.configure import (
    MAX_TOKENS_KEY,
    DEFAULT_MAX_TOKENS,
)

from .examples import AnnotatedJoinExample, JoinExample

DEFAULT_JOIN_FEW_SHOT: list[AnnotatedJoinExample] = [
    AnnotatedJoinExample(**d)
    for d in json.loads(
        open(Path(__file__).resolve().parent / "./default_examples.json", "r").read()
    )
]
MAIN_INSTRUCTION = "You are a database expert in charge of performing a modified `LEFT JOIN` operation. This `LEFT JOIN` is based on a semantic criteria given by the user.f\nIf a given left value has no corresponding right value, give '-' as a response. Stop responding as soon as all left values have been accounted for in the JSON mapping.\n"


@dataclass
class LLMJoin(JoinIngredient):
    DESCRIPTION = """
    If we need to do a `join` operation where there is imperfect alignment between table values, use the new function:
        `{{LLMJoin(left_on='table::column', right_on='table::column')}}`
    """
    model: Model = field(default=None)
    few_shot_retriever: Callable[[str], list[AnnotatedJoinExample]] = field(
        default=None
    )

    @classmethod
    def from_args(
        cls,
        model: Model | None = None,
        use_skrub_joiner: bool = True,
        few_shot_examples: list[dict] | list[AnnotatedJoinExample] | None = None,
        num_few_shot_examples: int | None = 1,
    ):
        """Creates a partial class with predefined arguments.

        Args:
            few_shot_examples: A list of AnnotatedJoinExamples dictionaries for few-shot learning.
                If not specified, will use [default_examples.json](https://github.com/parkervg/blendsql/blob/main/blendsql/ingredients/builtin/join/default_examples.json) as default.
            use_skrub_joiner: Whether to use the skrub joiner. Defaults to True.
            num_few_shot_examples: Determines number of few-shot examples to use for each ingredient call.
                Default is None, which will use all few-shot examples on all calls.
                If specified, will initialize a haystack-based DPR retriever to filter examples.

        Returns:
            Type[JoinIngredient]: A partial class of JoinIngredient with predefined arguments.

        Examples:
            ```python
            from blendsql import BlendSQL
            from blendsql.ingredients.builtin import LLMJoin, DEFAULT_JOIN_FEW_SHOT

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
                    num_few_shot_examples=2
                )
            }

            bsql = BlendSQL(db, ingredients=ingredients)
            ```
        """
        if few_shot_examples is None:
            few_shot_examples = DEFAULT_JOIN_FEW_SHOT
        else:
            few_shot_examples = [
                AnnotatedJoinExample(**d) if isinstance(d, dict) else d
                for d in few_shot_examples
            ]
        few_shot_retriever = initialize_retriever(
            examples=few_shot_examples, num_few_shot_examples=num_few_shot_examples
        )
        return cls._maybe_set_name_to_var_name(
            partialclass(
                cls,
                model=model,
                few_shot_retriever=few_shot_retriever,
                use_skrub_joiner=use_skrub_joiner,
            )
        )

    def run(
        self,
        model: Model,
        left_values: list[str],
        right_values: list[str],
        join_criteria: str | None = None,
        few_shot_retriever: Callable[[str], list[AnnotatedJoinExample]] = None,
        enable_constrained_decoding: bool = True,
        **kwargs,
    ) -> dict:
        """
        Args:
            model: The Model (blender) we will make calls to.
            left_values: List of values from the left table.
            right_values: List of values from the right table.
            join_criteria: Criteria for joining values.
            few_shot_retriever: Callable which takes a string, and returns n most similar few-shot examples.

        Returns:
            Dict mapping left values to right values.
        """
        if not enable_constrained_decoding:
            raise NotImplementedError(
                "Haven't implemented enable_constrained_decoding==False for LLMJoin yet"
            )

        if model is None:
            raise LMFunctionException(
                "LLMJoin requires a `Model` object, but nothing was passed!\nMost likely you forgot to set the `default_model` argument in `blend()`"
            )

        if join_criteria is None:
            join_criteria = "Join to same topics."
        else:
            join_criteria = dedent(join_criteria.removeprefix("\n"))

        if few_shot_retriever is None:
            # Default to 1 few-shot example in LLMJoin
            few_shot_retriever = lambda *_: DEFAULT_JOIN_FEW_SHOT[:1]

        current_example = JoinExample(
            **{
                "join_criteria": join_criteria,
                "left_values": sorted(left_values),
                "right_values": sorted(right_values),
            }
        )
        few_shot_examples: list[AnnotatedJoinExample] = few_shot_retriever(
            current_example.to_string()
        )

        if isinstance(model, ConstrainedModel):
            import guidance

            lm = LMString()

            @guidance(stateless=True, dedent=False)
            def make_predictions(lm, left_values, right_values):
                lm += "```json\n{"
                gen_f = guidance.select(options=right_values + ["-"])
                for idx, value in enumerate(left_values):
                    lm += (
                        f'\n\t"{value}": '
                        + '"'
                        + guidance.capture(gen_f, name=value)
                        + '"'
                        + ("," if idx + 1 != len(left_values) else "")
                    )  # Default to 1 few-shot example in LLMJoin

                lm += "\n}\n```"
                return lm

            curr_example_str = current_example.to_string()

            # First check - do we need to load the model?
            in_cache = False
            if model.caching:
                mapping, key = model.check_cache(
                    MAIN_INSTRUCTION,
                    curr_example_str,
                    "\n".join(
                        [
                            f"{example.to_string()}\n{example.mapping}"
                            for example in few_shot_examples
                        ]
                    ),
                    funcs=[make_predictions],
                )
                if mapping is not None:
                    in_cache = True
            model.prompt_tokens += len(
                model.tokenizer.encode(
                    MAIN_INSTRUCTION
                    + "\n".join(
                        [
                            f"{example.to_string()}{json.dumps(example.mapping, indent=4)}"
                            for example in few_shot_examples
                        ]
                    )
                    + curr_example_str
                )
            )
            if not in_cache:
                # Load our underlying guidance model, if we need to
                lm: guidance.models.Model = maybe_load_lm(model, lm)
                lm = model.maybe_add_system_prompt(lm)
                with guidance.user():
                    lm += MAIN_INSTRUCTION
                if len(few_shot_examples) > 0:
                    # Add few-shot examples
                    for example in few_shot_examples:
                        with guidance.user():
                            lm += example.to_string()
                        with guidance.assistant():
                            lm += (
                                "```json\n"
                                + json.dumps(example.mapping, indent=4)
                                + "\n```"
                            )
                with guidance.user():
                    lm += curr_example_str

                with guidance.assistant():
                    lm += make_predictions(
                        left_values=current_example.left_values,
                        right_values=current_example.right_values,
                    )  # type: ignore
                    model.num_generation_calls += 1
                mapping: dict = {
                    left_value: lm[left_value] for left_value in left_values
                }
                add_to_global_history(str(lm))
                if model.caching:
                    model.cache[key] = mapping  # type: ignore
            model.completion_tokens += sum(
                [len(model.tokenizer_encode(v)) for v in mapping.values()]  # type: ignore
            )

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
            "".join([i["content"] for i in messages])
            response = (
                model.generate(
                    messages_list=[messages],
                    max_tokens=kwargs.get(
                        "max_tokens", int(os.getenv(MAX_TOKENS_KEY, DEFAULT_MAX_TOKENS))
                    ),
                )[0]
                .removeprefix("```json")
                .removesuffix("```")
            )
            model.num_generation_calls += 1
            add_to_global_history(messages)
            # Post-process language model response
            try:
                mapping: dict = json.loads(response)
            except json.decoder.JSONDecodeError:
                mapping = {}
                logger.debug(
                    Color.error(
                        f"LLMJoin failed to return valid JSON!\nGot back '{response}'"
                    )
                )

        final_mapping = {k: v for k, v in mapping.items() if v != "-"}  # type: ignore
        logger.debug(
            Color.warning(
                f"Finished LLMJoin with values:\n{json.dumps({k: final_mapping[k] for k in list(final_mapping.keys())[:10]}, indent=4)}"
            )
        )
        return final_mapping
