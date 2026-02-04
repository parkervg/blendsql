import os
from typing import Callable
import json
from pathlib import Path
from dataclasses import dataclass, field
from textwrap import dedent
import asyncio
from guidance._grammar import select

from blendsql.configure import DEFAULT_ASYNC_LIMIT, ASYNC_LIMIT_KEY
from blendsql.models.model_base import ModelBase
from blendsql.common.logger import logger, Color
from blendsql.common.typing import GenerationResult, GenerationItem
from blendsql.ingredients.ingredient import JoinIngredient, LMFunctionException
from blendsql.ingredients.utils import initialize_retriever, partialclass

from .examples import AnnotatedJoinExample, JoinExample

DEFAULT_JOIN_FEW_SHOT: list[AnnotatedJoinExample] = [
    AnnotatedJoinExample(**d)
    for d in json.loads(
        open(Path(__file__).resolve().parent / "./default_examples.json", "r").read()
    )
]
MAIN_INSTRUCTION = "You are a database expert in charge of performing a modified `JOIN` operation. This `JOIN` is based on a semantic criteria given by the user.\nGiven the provided right values, generate a left value and give its alignment. If no alignment is present, use '-' as a placeholder for `NULL`.\n"


@dataclass
class LLMJoin(JoinIngredient):
    model: ModelBase = field(default=None)
    few_shot_retriever: Callable[[str], list[AnnotatedJoinExample]] = field(
        default=None
    )

    @classmethod
    def from_args(
        cls,
        model: ModelBase | None = None,
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

    async def run(
        self,
        model: ModelBase,
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

        n_parallel = (
            int(os.getenv(ASYNC_LIMIT_KEY, DEFAULT_ASYNC_LIMIT))
            if model._allows_parallel_requests
            else 1
        )
        cancel_event = asyncio.Event()
        semaphore = asyncio.Semaphore(n_parallel)
        left_values = sorted(list(left_values))
        right_values = sorted(list(right_values))
        current_example = JoinExample(
            **{
                "join_criteria": join_criteria,
                "left_values": left_values,
                "right_values": right_values,
            }
        )
        curr_example_str = current_example.to_string()
        few_shot_examples: list[AnnotatedJoinExample] = few_shot_retriever(
            curr_example_str
        )
        few_shot_str = "\n".join(
            [
                f"{example.to_string()}\n{example.mapping}"
                for example in few_shot_examples
            ]
        )

        mapping: dict[str, str] = {}
        base_prompt = MAIN_INSTRUCTION
        if len(few_shot_examples) > 0:
            for example in few_shot_examples:
                base_prompt += "\n" + example.to_string()
                base_prompt += (
                    "\n```json\n" + json.dumps(example.mapping, indent=4) + "\n```"
                )
        base_prompt += "\n" + curr_example_str

        grammar_prefix = '": '
        select_grammar = grammar_prefix + select(options=right_values + ["-"])

        items_to_process: list[GenerationItem] = []

        for left_value in left_values:
            cache_key = None
            if model.caching:
                cache_key = model.make_cache_key(
                    MAIN_INSTRUCTION,
                    curr_example_str,
                    few_shot_str,
                    left_value=left_value,
                )
                if cache_key in model.cache:
                    mapping[left_value] = model.cache[cache_key]
                    continue

            items_to_process.append(
                GenerationItem(
                    prompt=base_prompt,
                    assistant_continuation=f'\n```json\n{{\n\t"{left_value}"',
                    identifier=left_value,
                    cache_key=cache_key,
                    grammar=select_grammar.ll_grammar(),
                )
            )

        if not items_to_process:
            # All values were cached
            return {k: v for k, v in mapping.items() if v != "-"}

        # Track active tasks
        active_tasks: dict[asyncio.Task, GenerationItem] = {}
        items_submitted = 0
        items_completed = 0

        async def process_item(item: GenerationItem) -> GenerationResult | None:
            async with semaphore:
                if cancel_event.is_set():
                    return None

                # Generate with constrained grammar
                return await model.generate(
                    item,
                    cancel_event=cancel_event,
                )

        def submit_next_items():
            """Submit items up to n_parallel concurrency."""
            nonlocal items_submitted

            while (
                len(active_tasks) < n_parallel
                and items_submitted < len(items_to_process)
                and not cancel_event.is_set()
            ):
                item = items_to_process[items_submitted]
                task = asyncio.create_task(process_item(item))
                active_tasks[task] = item
                items_submitted += 1

        # Initial submission
        submit_next_items()

        # Process as tasks complete
        while active_tasks:
            done, _ = await asyncio.wait(
                active_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                item = active_tasks.pop(task)

                try:
                    result = task.result()
                except asyncio.CancelledError:
                    continue

                if result is None:
                    continue

                items_completed += 1

                mapping[result.identifier] = result.value.removeprefix(grammar_prefix)

                # Cache the result
                if model.caching and item.cache_key is not None:
                    model.cache[item.cache_key] = result.matched_value

            if cancel_event.is_set():
                break

            submit_next_items()

        model.num_generation_calls += items_completed

        final_mapping = {k: v for k, v in mapping.items() if v != "-"}
        logger.debug(
            Color.warning(
                f"Finished LLMJoin with values:\n{json.dumps({k: final_mapping[k] for k in list(final_mapping.keys())[:10]}, indent=4)}"
            )
        )
        return final_mapping
