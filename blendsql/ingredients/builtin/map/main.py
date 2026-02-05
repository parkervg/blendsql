import logging
import os
import asyncio
from typing import Callable, Generator
from pathlib import Path
import json
import polars as pl
import pandas as pd
from dataclasses import dataclass, field
from itertools import islice, repeat
from tqdm.auto import tqdm
from textwrap import indent, dedent
from guidance._grammar import select, gen
from guidance.library import substring

from blendsql.models.model_base import ModelBase
from blendsql.common.logger import logger, Color
from blendsql.common.constants import INDENT, DEFAULT_CONTEXT_FORMATTER
from blendsql.ingredients.ingredient import MapIngredient
from blendsql.common.exceptions import LMFunctionException
from blendsql.common.typing import (
    DataType,
    QuantifierType,
    AdditionalMapArg,
    GenerationItem,
    GenerationResult,
)
from blendsql.ingredients.utils import (
    initialize_retriever,
    partialclass,
    gen_list,
    _wrap_with_quotes,
)
from blendsql.configure import (
    MAX_OPTIONS_IN_PROMPT_KEY,
    DEFAULT_MAX_OPTIONS_IN_PROMPT,
    MAX_TOKENS_KEY,
    DEFAULT_MAX_TOKENS,
    ASYNC_LIMIT_KEY,
    DEFAULT_ASYNC_LIMIT,
)
from blendsql.types import prepare_datatype, apply_type_conversion
from .examples import (
    MapExample,
    AnnotatedMapExample,
    FeatureType,
)
from blendsql.search.searcher import Searcher

DEFAULT_MAP_FEW_SHOT: list[AnnotatedMapExample] = [
    AnnotatedMapExample(**d)
    for d in json.loads(
        open(Path(__file__).resolve().parent / "./default_examples.json", "r").read()
    )
]
CONSTRAINED_MAIN_INSTRUCTION = "Complete the docstring for the provided Python function. The output will correctly answer the question provided for each input value, with no mistakes. "
CONSTRAINED_MAIN_INSTRUCTION = (
    CONSTRAINED_MAIN_INSTRUCTION
    + "On each newline, you will follow the format of f({value}) == {answer}.\n"
)


@dataclass
class LLMMap(MapIngredient):
    model: ModelBase = field(default=None)
    few_shot_retriever: Callable[[str], list[AnnotatedMapExample]] = field(default=None)
    list_options_in_prompt: bool = field(default=True)
    few_shot_retriever: Callable[[str], list[AnnotatedMapExample]] = field(default=None)
    context_formatter: Callable[[pl.DataFrame], str] = field(
        default=DEFAULT_CONTEXT_FORMATTER,
    )
    batch_size: int = field(default=None)

    @classmethod
    def from_args(
        cls,
        model: ModelBase | None = None,
        few_shot_examples: list[dict] | list[AnnotatedMapExample] | None = None,
        list_options_in_prompt: bool = True,
        options_searcher: Searcher | None = None,
        batch_size: int | None = None,
        num_few_shot_examples: int | None = 0,
        context_searcher: Searcher | None = None,
    ):
        """Creates a partial class with predefined arguments.

        Args:
            model: The model to be used. Defaults to None.
            few_shot_examples: A list of dictionary MapExample few-shot examples.
               If not specified, will use [default_examples.json](https://github.com/parkervg/blendsql/blob/main/blendsql/ingredients/builtin/map/default_examples.json) as default.
            list_options_in_prompt: Whether to list options in the prompt. Defaults to True.
            options_searcher: A callable that takes in a list of options, and returns a `Searcher` object.
                For example, ```
                options_searcher=lambda d: HybridSearch(
                    documents=d,
                    model_name_or_path="intfloat/e5-base-v2",
                    k=10,
                )
                ```
            batch_size: The batch size for processing. Defaults to 1.
            num_few_shot_examples: Determines number of few-shot examples to use for each ingredient call.
               Default is None, which will use all few-shot examples on all calls.
               If specified, will initialize a haystack-based embedding retriever to filter examples.

        Returns:
            Type[MapIngredient]: A partial class of MapIngredient with predefined arguments.

        Examples:
            ```python
            from blendsql import BlendSQL
            from blendsql.ingredients.builtin import LLMQA, DEFAULT_QA_FEW_SHOT

            ingredients = {
                LLMQA.from_args(
                    few_shot_examples=[
                        *DEFAULT_QA_FEW_SHOT,
                        {
                            "question": "Which weighs the most?",
                            "context": {
                                {
                                    "Animal": ["Dog", "Gorilla", "Hamster"],
                                    "Weight": ["20 pounds", "350 lbs", "100 grams"]
                                }
                            },
                            "answer": "Gorilla",
                            # Below are optional
                            "options": ["Dog", "Gorilla", "Hamster"]
                        }
                    ],
                    num_few_shot_examples=2,
                    # Lambda to turn the pd.DataFrame to a serialized string
                    context_formatter=lambda df: df.to_markdown(
                        index=False
                    )
                )
            }

            bsql = BlendSQL(db, ingredients=ingredients)
            ```
        """
        if few_shot_examples is None:
            few_shot_examples = DEFAULT_MAP_FEW_SHOT
        few_shot_retriever = initialize_retriever(
            examples=few_shot_examples, num_few_shot_examples=num_few_shot_examples
        )
        return cls._maybe_set_name_to_var_name(
            partialclass(
                cls,
                model=model,
                few_shot_retriever=few_shot_retriever,
                list_options_in_prompt=list_options_in_prompt,
                options_searcher=options_searcher,
                batch_size=batch_size,
                context_searcher=context_searcher,
            )
        )

    async def run(
        self,
        model: ModelBase,
        question: str,
        values: list[str],
        additional_args: list[AdditionalMapArg],
        list_options_in_prompt: bool,
        context_formatter: Callable[[pl.DataFrame], str],
        global_subtable_context: pl.DataFrame | None = None,
        context_searcher: Searcher | None = None,
        unpacked_questions: list[str] = None,
        options: list[str] | None = None,
        options_searcher: Searcher | None = None,
        few_shot_retriever: Callable[[str], list[AnnotatedMapExample]] = None,
        value_limit: int | None = None,
        example_outputs: str | None = None,
        quantifier: QuantifierType = None,
        return_type: DataType | str | None = None,
        regex: str | None = None,
        batch_size: int = None,
        exit_condition_func: Callable = None,
        exit_condition_required_values: int = None,
        enable_constrained_decoding: bool = True,
        **kwargs,
    ) -> list[float | int | str | bool]:
        """For each value in a given column, calls a Model and retrieves the output.

        Args:
            question: The question(s) to map onto the values. Will also be the new column name
            model: The Model (blender) we will make calls to.
            values: The list of values to apply question to.
            value_limit: Optional limit on the number of values to pass to the Model
            example_outputs: This gives the Model an example of the output we expect.
            return_type: In the absence of example_outputs, give the Model some signal as to what we expect as output.
            regex: Optional regex to constrain answer generation.

        Returns:
            Iterable[Any] containing the output of the Model for each value.
        """
        if model is None:
            raise LMFunctionException(
                "LLMMap requires a `Model` object, but nothing was passed!\nMost likely you forgot to set the `default_model` argument in `blend()`"
            )
        if few_shot_retriever is None:
            few_shot_retriever = lambda *_: []

        question = dedent(question.removeprefix("\n"))

        context: list[str | None] = [None] * len(values)
        context_in_use_type: FeatureType = None

        if context_searcher is not None and global_subtable_context is None:
            if (
                unpacked_questions is not None
            ):  # Implies we have different context for each value
                context = context_searcher(unpacked_questions)
                context_in_use_type = FeatureType.LOCAL
            else:
                context = " | ".join(context_searcher(question)[0])
                context_in_use_type = FeatureType.GLOBAL
        elif global_subtable_context is not None:
            context = context_formatter(global_subtable_context)
            context_in_use_type = FeatureType.GLOBAL

        # Log what we found
        if context_in_use_type == FeatureType.GLOBAL:
            logger.debug(
                Color.quiet_update(f"Retrieved global context '{context[:50]}...'")
            )
        elif context_in_use_type == FeatureType.LOCAL:
            logger.debug(
                Color.quiet_update(
                    f"Retrieved local contexts '{[str(d[:2]) + '...' for d in context[:3]]}...'"
                )
            )
        elif context_in_use_type is not None:
            raise ValueError(
                f"Invalid `context_in_use_type`: {type(context_in_use_type)}"
            )

        # Unpack default kwargs
        table_name, column_name = self.unpack_default_kwargs(**kwargs)
        if value_limit is not None:
            values = values[:value_limit]
        values = [str(value) if not pd.isna(value) else "-" for value in values]
        resolved_return_type: DataType = prepare_datatype(
            return_type=return_type, options=options, quantifier=quantifier
        )

        # Prep options, if passed
        if self.options_searcher is None:
            options_in_use_type = FeatureType.GLOBAL
            if options is not None and list_options_in_prompt:
                max_options_in_prompt = int(
                    os.getenv(MAX_OPTIONS_IN_PROMPT_KEY, DEFAULT_MAX_OPTIONS_IN_PROMPT)
                )
                if len(options) > max_options_in_prompt:
                    logger.debug(
                        Color.warning(
                            f"Number of options ({len(options):,}) is greater than the configured MAX_OPTIONS_IN_PROMPT={max_options_in_prompt:,}.\nWill run inference without explicitly listing these options in the prompt text."
                        )
                    )
                    list_options_in_prompt = False
        else:
            logger.debug(
                Color.warning(
                    f"Calling provided `options_searcher` to retrieve {self.options_searcher.k} options for each of the {len(values)} unique values, out of {len(self.options_searcher.documents):,} total options..."
                )
            )
            options_in_use_type = FeatureType.LOCAL

        filtered_options: list[str | None] = [None] * len(values)
        if self.options_searcher is not None:
            if context_in_use_type is not None:
                documents = [f"{v} | {c}" for c, v in zip(values, context)]
            else:
                documents = values
            filtered_options = self.options_searcher(documents)

        current_example = MapExample(
            question=question,
            column_name=column_name,
            table_name=table_name,
            return_type=resolved_return_type,
            example_outputs=example_outputs,
            options_type=options_in_use_type,
            options=options,
            context_type=context_in_use_type,
            context=context,
        )

        current_example = MapExample(**current_example.__dict__)
        few_shot_examples: list[AnnotatedMapExample] = [
            AnnotatedMapExample(**example.__dict__)
            if not isinstance(example, dict)
            else AnnotatedMapExample(**example)
            for example in few_shot_retriever(
                current_example.to_string(
                    additional_args=additional_args,
                )
            )
        ]

        is_list_output = resolved_return_type.quantifier is not None
        regex = regex or resolved_return_type.regex
        quantifier = resolved_return_type.quantifier

        if all(x is not None for x in [options, regex]):
            raise LMFunctionException(
                "MapIngredient exception!\nCan't have both `options` and `regex` argument passed."
            )

        n_parallel = (
            int(os.getenv(ASYNC_LIMIT_KEY, DEFAULT_ASYNC_LIMIT))
            if model._allows_parallel_requests
            else 1
        )

        gen_f = None
        grammar_prefix = " == "
        grammar_suffix = f"\n"
        if enable_constrained_decoding:
            if is_list_output:
                if self.options_searcher is not None:
                    # Need to create separate gen_f for each set of filtered_options
                    gen_f = [
                        lambda _, o=o: grammar_prefix
                        + gen_list(
                            force_quotes=resolved_return_type.requires_quotes,
                            quantifier=quantifier,
                            options=o,
                            regex=regex,
                        )
                        + grammar_suffix
                        for o in filtered_options
                    ]
                else:
                    gen_f = (
                        lambda _: grammar_prefix
                        + gen_list(
                            force_quotes=resolved_return_type.requires_quotes,
                            quantifier=quantifier,
                            options=options,
                            regex=regex,
                        )
                        + grammar_suffix
                    )
            else:
                if self.options_searcher is not None:
                    # Need to create separate gen_f for each set of filtered_options
                    gen_f = [
                        lambda _, o=o: grammar_prefix
                        + _wrap_with_quotes(
                            select(options=o),
                            has_options_or_regex=bool(o or regex),
                            force_quotes=resolved_return_type.requires_quotes,
                        )
                        + grammar_suffix
                        for o in filtered_options
                    ]
                elif options is not None:
                    select_fn = select(options=options)
                    gen_f = (
                        lambda _: grammar_prefix
                        + _wrap_with_quotes(
                            select_fn,
                            has_options_or_regex=bool(options or regex),
                            force_quotes=resolved_return_type.requires_quotes,
                        )
                        + grammar_suffix
                    )
                elif resolved_return_type.name == "substring":
                    # Special case for substring datatypes
                    gen_f = (
                        lambda s: grammar_prefix
                        + _wrap_with_quotes(
                            substring(target_string=s),
                            has_options_or_regex=bool(options or regex),
                            force_quotes=resolved_return_type.requires_quotes,
                        )
                        + grammar_suffix
                    )
        else:
            logger.debug(
                Color.warning(
                    "Not applying constraints, since `enable_constrained_decoding==False`"
                )
            )
        if gen_f is None:
            # Create base gen_f function
            gen_f = lambda _: grammar_prefix + _wrap_with_quotes(
                gen(
                    max_tokens=kwargs.get(
                        "max_tokens",
                        int(os.getenv(MAX_TOKENS_KEY, DEFAULT_MAX_TOKENS)),
                    ),
                    # guidance>=0.2.1 doesn't allow both `stop` and `regex` to be passed
                    # guidance 0.3.0 raises a serialization error if this is a list, not a tuple
                    stop=None if regex is not None else '"\n',
                    regex=regex if enable_constrained_decoding else None,
                )
                + grammar_suffix,
                has_options_or_regex=bool(options or regex),
                force_quotes=resolved_return_type.requires_quotes,
            )

        def format_current_prompt(
            value: str,
            additional_args: tuple[str] | None,
            context: str | list[str] | None,
            context_in_use_type: FeatureType | None,
            local_options: list[str] | None,
        ) -> str:
            def get_quote(s: str):
                return '"""' if any(c in s for c in ["\n", '"']) else '"'

            value_quote = get_quote(value)
            has_more_than_one_arg = bool(
                context_in_use_type == FeatureType.LOCAL
                or additional_args is not None
                or local_options is not None
            )
            arg_prefix = " "
            newline_args = False
            if has_more_than_one_arg:
                newline_args = False
                if len(value) > 20:
                    newline_args = True
                elif local_options is not None:
                    if len(str(local_options)) > 20:
                        newline_args = True
                elif additional_args is not None:
                    for a in additional_args:
                        if len(a) > 20:
                            newline_args = True
                if newline_args:
                    arg_prefix = f"\n{INDENT(3)}"  # If we pass more than one arg, and they are long, make them appear on newlines
                gen_str = f"""{INDENT(2)}f({arg_prefix if newline_args else ''}{value_quote}{value}{value_quote}"""
                if additional_args is not None:
                    for arg in additional_args:
                        gen_str += f',{arg_prefix}"{arg}"'
                if context_in_use_type == FeatureType.LOCAL:
                    json_str = json.dumps(context, ensure_ascii=False, indent=16)[:-1]
                    gen_str += f",{arg_prefix}" + json_str + f"{INDENT(3)}]"
                if local_options is not None:
                    gen_str += f",{arg_prefix}{local_options}"
            else:  # Global contexts have already been handled. We only have a single variable to pass.
                indented_value = value.replace("\n", f"\n{INDENT(2)}")
                gen_str = f"""{INDENT(2)}f({value_quote}{indented_value}{value_quote}"""
            if has_more_than_one_arg:
                if newline_args:
                    gen_str += f"\n{INDENT(2)}"
            gen_str += ")"
            return gen_str

        example_str = ""
        if len(few_shot_examples) > 0:
            for example in few_shot_examples:
                example_str += example.to_string()

        lm_mapping = {}  # Where we store the final type-cast results

        curr_function_signature_str = current_example.to_string(
            list_options=list_options_in_prompt,
            additional_args=additional_args,
            add_leading_newlines=True,
        )

        base_prompt = CONSTRAINED_MAIN_INSTRUCTION
        if example_str:
            base_prompt += example_str
            base_prompt += "\n\nNow, complete the docstring for the following example:"
        base_prompt += curr_function_signature_str

        lm_mapping: dict = {}  # Final type-cast results
        all_processed_identifiers: list[str] = []

        def generate_items() -> Generator[GenerationItem, None, None]:
            """Lazily yields items, checking cache as we go."""
            for idx, (v, a, c, o) in enumerate(
                zip(
                    values,
                    zip(*[arg.values for arg in additional_args])
                    if additional_args
                    else repeat(None),
                    context,
                    filtered_options,
                )
            ):
                # Build identifier
                curr_identifier = v
                if a is not None:
                    curr_identifier += f"_{a}"
                if c is not None:
                    curr_identifier += f"_{c}"

                all_processed_identifiers.append(curr_identifier)

                # Check cache
                cache_key = None
                if model.caching:
                    cached_response, cache_key = model.check_cache(
                        CONSTRAINED_MAIN_INSTRUCTION,
                        example_str,
                        question,
                        regex,
                        options,
                        quantifier,
                        kwargs.get(
                            "max_tokens",
                            int(os.getenv(MAX_TOKENS_KEY, DEFAULT_MAX_TOKENS)),
                        ),
                        v,
                        a,
                        c,
                        o,
                        funcs=[
                            format_current_prompt,
                            gen_f[idx]
                            if hasattr(gen_f, "__getitem__")
                            and enable_constrained_decoding
                            else gen_f,
                        ],
                    )
                    if cached_response is not None:
                        lm_mapping[curr_identifier] = cached_response
                        continue  # Skip - already cached

                # Build prompt
                assistant_continuation = format_current_prompt(
                    value=v,
                    additional_args=a,
                    context=c,
                    context_in_use_type=context_in_use_type,
                    local_options=o,
                )

                # Get grammar
                if hasattr(gen_f, "__getitem__") and enable_constrained_decoding:
                    # Different grammars for each item
                    grammar = gen_f[idx](v).ll_grammar()
                else:
                    grammar = gen_f(v).ll_grammar()

                yield GenerationItem(
                    identifier=curr_identifier,
                    prompt=base_prompt,
                    assistant_continuation=assistant_continuation,
                    grammar=grammar,
                    cache_key=cache_key,
                )

        cancel_event = asyncio.Event()
        semaphore = asyncio.Semaphore(n_parallel)
        n_satisfied = 0
        items_submitted = 0
        items_completed = 0

        # Track active tasks and their items
        active_tasks: dict[asyncio.Task, GenerationItem] = {}
        item_generator = generate_items()
        generator_exhausted = False

        async def process_item(item: GenerationItem) -> GenerationResult | None:
            async with semaphore:
                if cancel_event.is_set():
                    return None
                self.num_values_passed += 1
                return await model.generate(item, cancel_event)

        def submit_next_items():
            """Submit items up to n_parallel concurrency."""
            nonlocal generator_exhausted, items_submitted

            while (
                len(active_tasks) < n_parallel
                and not generator_exhausted
                and not cancel_event.is_set()
            ):
                try:
                    item = next(item_generator)
                    task = asyncio.create_task(process_item(item))
                    active_tasks[task] = item
                    items_submitted += 1
                except StopIteration:
                    generator_exhausted = True
                    break

        if logger.level <= logging.DEBUG:
            pbar = tqdm(
                desc=Color.prefix + f"LLMMap with n_parallel={n_parallel}",
                total=len(values),
            )
            # Update for cached items
            pbar.update(len(lm_mapping))

        # Initial submission
        submit_next_items()

        # Process as tasks complete
        while active_tasks:
            # Wait for at least one task to complete
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

                # Type conversion
                converted_value = apply_type_conversion(
                    result.value.removeprefix(grammar_prefix).split(grammar_suffix)[0],
                    return_type=resolved_return_type,
                    db=self.db,
                )

                lm_mapping[result.identifier] = converted_value

                # Cache result
                if model.caching and item.cache_key is not None:
                    model.cache[item.cache_key] = converted_value

                if logger.level <= logging.DEBUG:
                    pbar.update(1)

                # Check exit condition
                if exit_condition_func and result.completed:
                    if exit_condition_func(converted_value):
                        n_satisfied += 1

                        if n_satisfied >= exit_condition_required_values:
                            logger.debug(
                                Color.optimization(
                                    f"[ 🚪] Exit condition satisfied. Exiting early after processing {items_completed:,} out of {items_submitted:,} items, {len(lm_mapping)} total (including cached)."
                                )
                            )
                            cancel_event.set()

                            # Cancel pending tasks
                            for t in active_tasks:
                                t.cancel()

                            # Wait for cancellations
                            if active_tasks:
                                await asyncio.gather(
                                    *active_tasks.keys(), return_exceptions=True
                                )
                            active_tasks.clear()
                            break
            if cancel_event.is_set():
                break

            submit_next_items()

        if logger.level <= logging.DEBUG:
            pbar.close()

        mapped_values = [
            lm_mapping.get(identifier, None) for identifier in all_processed_identifiers
        ]
        # Find difference in length, and fill `None`
        mapped_values.extend([None] * (len(values) - len(all_processed_identifiers)))

        logger.debug(
            lambda: Color.warning(
                f"Finished LLMMap with {len(lm_mapping)} total values{' (10 shown)' if len(lm_mapping) > 10 else ''}:\n{indent(json.dumps({str(k): str(v) for k, v in islice(lm_mapping.items(), 10)}, indent=4), Color.prefix if Color.in_block else '')}"
            )
        )
        return mapped_values
