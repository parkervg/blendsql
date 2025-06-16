import typing as t
import json
from colorama import Fore
from pathlib import Path
from attr import attrs, attrib

from blendsql.models import Model, ConstrainedModel
from blendsql.models.constrained.utils import LMString, maybe_load_lm
from blendsql.common.logger import logger
from blendsql.ingredients.ingredient import JoinIngredient, IngredientException
from blendsql.ingredients.utils import initialize_retriever, partialclass

from .examples import AnnotatedJoinExample, JoinExample

DEFAULT_JOIN_FEW_SHOT: t.List[AnnotatedJoinExample] = [
    AnnotatedJoinExample(**d)
    for d in json.loads(
        open(Path(__file__).resolve().parent / "./default_examples.json", "r").read()
    )
]
MAIN_INSTRUCTION = "You are a database expert in charge of performing a modified `LEFT JOIN` operation. This `LEFT JOIN` is based on a semantic criteria given by the user.f\nIf a given left value has no corresponding right value, give '-' as a response. Stop responding as soon as all left values have been accounted for in the JSON mapping.\n"


@attrs
class LLMJoin(JoinIngredient):
    DESCRIPTION = """
    If we need to do a `join` operation where there is imperfect alignment between table values, use the new function:
        `{{LLMJoin(left_on='table::column', right_on='table::column')}}`
    """
    model: Model = attrib(default=None)
    few_shot_retriever: t.Callable[[str], t.List[AnnotatedJoinExample]] = attrib(
        default=None
    )

    @classmethod
    def from_args(
        cls,
        model: t.Optional[Model] = None,
        use_skrub_joiner: bool = True,
        few_shot_examples: t.Optional[
            t.Union[t.List[dict], t.List[AnnotatedJoinExample]]
        ] = None,
        num_few_shot_examples: t.Optional[int] = None,
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
        left_values: t.List[str],
        right_values: t.List[str],
        join_criteria: t.Optional[str] = None,
        few_shot_retriever: t.Callable[[str], t.List[AnnotatedJoinExample]] = None,
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
        if model is None:
            raise IngredientException(
                "LLMJoin requires a `Model` object, but nothing was passed!\nMost likely you forgot to set the `default_model` argument in `blend()`"
            )

        if join_criteria is None:
            join_criteria = "Join to same topics."
        if few_shot_retriever is None:
            few_shot_retriever = lambda *_: DEFAULT_JOIN_FEW_SHOT

        current_example = JoinExample(
            **{
                "join_criteria": join_criteria,
                "left_values": sorted(left_values),
                "right_values": sorted(right_values),
            }
        )
        few_shot_examples: t.List[AnnotatedJoinExample] = few_shot_retriever(
            current_example.to_string()
        )

        if isinstance(model, ConstrainedModel):
            import guidance

            lm = LMString()

            @guidance(stateless=True, dedent=False)
            def make_predictions(lm, left_values, right_values):
                lm += "```json\n{"
                gen_f = guidance.select(options=right_values)
                for idx, value in enumerate(left_values):
                    lm += (
                        f'\n\t"{value}": '
                        + guidance.capture(gen_f, name=value)
                        + ("," if idx + 1 != len(left_values) else "")
                    )
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
                model.num_generation_calls += 1
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
                mapping: dict = lm._variables
                # mapping: dict = {k: v['value'] for k, v in lm._state.captures.items()}
                if model.caching:
                    model.cache[key] = mapping  # type: ignore
            model.completion_tokens += sum(
                [len(model.tokenizer.encode(v)) for v in mapping.values()]  # type: ignore
            )

        else:
            # Use DSPy to get LLM output
            import dspy

            join_fn = dspy.Predict(
                dspy.Signature(
                    f"left_values: List[str], right_values: List[str], join_criteria: str -> mapping: Dict[str, str]",
                    instructions=MAIN_INSTRUCTION,
                )
            )
            join_fn.demos = [
                dspy.Example(**example.__dict__) for example in few_shot_examples
            ]

            signature = join_fn.dump_state()["signature"]
            fn_kwargs = {
                "join_criteria": join_criteria,
                "left_values": current_example.left_values,
                "right_values": current_example.right_values,
            }

            # First check - do we need to load the model?
            in_cache = False
            if model.caching:
                cached_response_data, cache_key = model.check_cache(
                    signature, fn_kwargs
                )
                if cached_response_data is not None:
                    mapping, token_stats = (
                        cached_response_data["response"],
                        cached_response_data["token_stats"],
                    )
                    prompt_tokens = token_stats["prompt_tokens"]
                    completion_tokens = token_stats["completion_tokens"]
                    in_cache = True

            if not in_cache:
                # Generate the response
                mapping = model.generate(
                    join_fn,
                    kwargs_list=[fn_kwargs],
                )[0].mapping
                model.num_generation_calls += 1

                # Get token usage if available
                prompt_tokens, completion_tokens = model.get_token_usage(1)

                # Store in cache if caching is enabled
                if model.caching:
                    model.cache[cache_key] = {
                        "response": mapping,
                        "token_stats": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        },
                    }

                model.completion_tokens += completion_tokens
                model.prompt_tokens += prompt_tokens

        final_mapping = {k: v for k, v in mapping.items() if v != "-"}  # type: ignore
        logger.debug(
            Fore.YELLOW
            + f"Finished LLMJoin with values:\n{json.dumps({k: final_mapping[k] for k in list(final_mapping.keys())[:10]}, indent=4)}"
            + Fore.RESET
        )
        return final_mapping
