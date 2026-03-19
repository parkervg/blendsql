import logging
import os
import asyncio
from typing import Callable, Generator, Literal
import json
import polars as pl
import pandas as pd
from dataclasses import dataclass, field
from itertools import islice, repeat
from tqdm.auto import tqdm
from textwrap import indent, dedent
from guidance import json as guidance_json
from pydantic import TypeAdapter
from guidance.library import substring
from guidance import regex as guidance_regex

from blendsql.models.model_base import ModelBase
from blendsql.common.logger import logger, Color
from blendsql.common.constants import DEFAULT_CONTEXT_FORMATTER
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
    partialclass,
    gen_list,
    _wrap_with_quotes,
    get_python_type,
    parse_quantifier,
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
from blendsql.search.searcher import Searcher
from .prompts import (
    FeatureType,
    BASE_RETURN_TYPE_TO_EXAMPLE,
    BASE_RETURN_TYPE_TO_INSTRUCTION,
)


@dataclass
class LLMMap(MapIngredient):
    model: ModelBase = field(default=None)
    list_options_in_prompt: bool = field(default=True)
    context_formatter: Callable[[pl.DataFrame], str] = field(
        default=DEFAULT_CONTEXT_FORMATTER,
    )
    prompt_style: Literal["basic", "python"] = "basic"

    @classmethod
    def from_args(
        cls,
        model: ModelBase | None = None,
        return_type_to_example: list[dict] | None = None,
        list_options_in_prompt: bool = True,
        options_searcher: Searcher | None = None,
        context_searcher: Searcher | None = None,
        prompt_style: Literal["basic", "python"] = "basic",
    ):
        """Creates a partial class with predefined arguments.

        Args:
            model: The model to be used. Defaults to None.
            list_options_in_prompt: Whether to list options in the prompt. Defaults to True.
            options_searcher: A callable that takes in a list of options, and returns a `Searcher` object.
                For example, ```
                options_searcher=lambda d: HybridSearch(
                    documents=d,
                    model_name_or_path="intfloat/e5-base-v2",
                    k=10,
                )
                ```
            prompt_style: One of 'python' or 'xml'. Controls the prompt format sent to the model.

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
        return cls._maybe_set_name_to_var_name(
            partialclass(
                cls,
                model=model,
                list_options_in_prompt=list_options_in_prompt,
                options_searcher=options_searcher,
                return_type_to_example=return_type_to_example,
                context_searcher=context_searcher,
                prompt_style=prompt_style,
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
        value_limit: int | None = None,
        example_outputs: str | None = None,
        quantifier: QuantifierType = None,
        return_type: DataType | str | None = None,
        regex: str | None = None,
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

        if unpacked_questions is None:
            unpacked_questions = [None] * len(values)

        assert len(unpacked_questions) == len(values)

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

        is_list_output = resolved_return_type.quantifier is not None
        regex = regex or resolved_return_type.regex
        quantifier = resolved_return_type.quantifier

        quantifier_min_length, quantifier_max_length = parse_quantifier(quantifier)

        if all(x is not None for x in [options, regex]):
            raise LMFunctionException(
                "MapIngredient exception!\nCan't have both `options` and `regex` argument passed."
            )

        n_parallel = int(os.getenv(ASYNC_LIMIT_KEY, DEFAULT_ASYNC_LIMIT))

        grammar = None
        grammar_suffix = "\n"
        if enable_constrained_decoding:
            if is_list_output:
                if self.options_searcher is not None:
                    # Need to create separate grammar for each set of filtered_options
                    grammar = [
                        lambda _, o=o: (
                            gen_list(
                                data_type=resolved_return_type,
                                quantifier=quantifier,
                                options=o,
                                quantifier_min_length=quantifier_min_length,
                                quantifier_max_length=quantifier_max_length,
                            )
                            + grammar_suffix
                        ).ll_grammar()
                        for o in filtered_options
                    ]
                else:
                    grammar = lambda _: (
                        gen_list(
                            data_type=resolved_return_type,
                            quantifier=quantifier,
                            options=options,
                            quantifier_min_length=quantifier_min_length,
                            quantifier_max_length=quantifier_max_length,
                        )
                        + grammar_suffix
                    ).ll_grammar()
            else:
                if self.options_searcher is not None:
                    # Need to create separate grammar for each set of filtered_options
                    grammar = [
                        lambda _, o=o: (
                            guidance_json(
                                schema=TypeAdapter(get_python_type(options=o))
                            )
                            + grammar_suffix
                        ).ll_grammar()
                        for o in filtered_options
                    ]
                elif resolved_return_type.name == "substring":
                    # Special case for substring datatypes
                    # This can't be written as a `guidance_json` obj
                    grammar = lambda s: (
                        _wrap_with_quotes(
                            substring(target_string=s),
                            has_options_or_regex=bool(options or regex),
                            force_quotes=resolved_return_type.requires_quotes
                            and self.prompt_style == "python",
                        )
                        + grammar_suffix
                    ).ll_grammar()
                elif regex is not None:
                    # pydantic TypeAdapters don't work here
                    grammar = lambda _: (
                        _wrap_with_quotes(
                            guidance_regex(pattern=regex),
                            has_options_or_regex=bool(options or regex),
                            force_quotes=resolved_return_type.requires_quotes
                            and self.prompt_style == "python",
                        )
                        + grammar_suffix
                    ).ll_grammar()
        else:
            logger.debug(
                Color.warning(
                    "Not applying constraints, since `enable_constrained_decoding==False`"
                )
            )
        if grammar is None and resolved_return_type.name != "str":
            # Create base grammar function
            grammar = lambda _: (
                guidance_json(
                    schema=TypeAdapter(
                        get_python_type(
                            data_type=resolved_return_type,
                            options=options,
                        )
                    ),
                    max_tokens=kwargs.get(
                        "max_tokens", int(os.getenv(MAX_TOKENS_KEY, DEFAULT_MAX_TOKENS))
                    ),
                )
                + grammar_suffix
            ).ll_grammar()

        return_type_to_example = BASE_RETURN_TYPE_TO_EXAMPLE | (
            self.return_type_to_example or dict()
        )
        if options is not None:
            key = "literal"
        else:
            key = resolved_return_type.name.lower()
        instruction: str = BASE_RETURN_TYPE_TO_INSTRUCTION.get(
            key, BASE_RETURN_TYPE_TO_INSTRUCTION["str"]
        )
        if resolved_return_type.quantifier is not None:
            # Add a note about how many items we expect back
            if quantifier == "+":
                instruction += (
                    " There should be one or more items in your generated list."
                )
            elif quantifier == "*":
                instruction += (
                    " There should be zero or more items in your generated list."
                )
            elif quantifier_max_length == quantifier_min_length:
                instruction += f" There should be exactly {quantifier_max_length} items in your generated list."
            else:
                instruction += f" There should be between {quantifier_min_length} and {quantifier_max_length} items in your generated list."
        instruction += " An example is shown below."
        one_shot_example: dict = return_type_to_example.get(
            key, return_type_to_example["str"]
        )

        if self.prompt_style == "python":
            from .prompts import (
                format_python_signature,
                format_python_continuation,
                PYTHON_INSTRUCTION,
            )

            prompt = PYTHON_INSTRUCTION
            example_signature = format_python_signature(
                question=one_shot_example["question"],
                return_type=resolved_return_type,
                context=one_shot_example.get("context"),
                context_type=FeatureType.GLOBAL
                if one_shot_example.get("context")
                else None,
                options=one_shot_example.get("options"),
                list_options=True,
                options_type=FeatureType.GLOBAL,
            )
            example_string = ""
            for example in one_shot_example["examples"]:
                example_string += format_python_continuation(
                    value=example["value"],
                    additional_args=example.get("additional_args", None),
                    context=example.get("context", None),
                    local_options=example.get("options", None),
                    context_in_use_type=FeatureType.LOCAL
                    if example.get("context")
                    else None,
                )
                answer = example["answer"]
                example_string += (
                    f'"{answer}"' if isinstance(answer, str) else f"{answer}"
                )

            prompt += example_signature + example_string + "\n```"
            prompt += "\n\nNow, complete the docstring for the following example:"
            curr_function_signature_str = format_python_signature(
                question=question,
                column_name=column_name,
                table_name=table_name,
                return_type=resolved_return_type,
                options_type=options_in_use_type,
                options=options,
                context_type=context_in_use_type,
                context=context,
                list_options=list_options_in_prompt,
                additional_args=additional_args,
                add_leading_newlines=True,
            )
            prompt += curr_function_signature_str

        elif self.prompt_style == "basic":
            from .prompts import format_default_continuation

            prompt = instruction
            example_string = ""
            for idx, example in enumerate(one_shot_example["examples"]):
                example_string += format_default_continuation(
                    question=one_shot_example["question"] if idx == 0 else None,
                    value=example["value"],
                    additional_args=example.get("additional_args", None),
                    context=one_shot_example.get("context", None)
                    or example.get("context"),
                    options=one_shot_example.get("options", None)
                    or example.get("options", None),
                )
                if not isinstance(example["answer"], str):
                    example_string += (
                        json.dumps(example["answer"], ensure_ascii=False) + "\n\n"
                    )
                else:
                    example_string += example["answer"] + "\n\n"
            prompt = f"{instruction}\n\n{example_string}" + "---\n\n"
        else:
            raise ValueError(
                f"Unknown prompt style: {self.prompt_style}\nValid arguments are ['python', 'basic']"
            )
        lm_mapping: dict = {}  # Final type-cast results
        all_processed_identifiers: list[str] = []

        def generate_items() -> Generator[GenerationItem, None, None]:
            """Lazily yields items, checking cache as we go."""
            grammar_is_collection = hasattr(grammar, "__getitem__")
            for idx, (q, v, a, c, o) in enumerate(
                zip(
                    unpacked_questions,
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
                        prompt,
                        self.prompt_style,
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
                            grammar[idx]
                            if grammar_is_collection and enable_constrained_decoding
                            else grammar,
                        ]
                        if grammar is not None
                        else None,
                    )
                    if cached_response is not None:
                        lm_mapping[curr_identifier] = cached_response
                        continue  # Skip - already cached

                # Build per-item prompt/continuation.
                if self.prompt_style == "python":
                    item_prompt = prompt + format_python_continuation(
                        value=v,
                        additional_args=a,
                        context=c,
                        context_in_use_type=context_in_use_type,
                        local_options=o,
                    )
                    assistant_continuation = None
                elif self.prompt_style == "basic":
                    _context = None
                    if context_in_use_type == FeatureType.GLOBAL:
                        _context = context
                    elif context_in_use_type == FeatureType.LOCAL:
                        _context = c
                    item_prompt = prompt + format_default_continuation(
                        value=v,
                        additional_args=a,
                        context=_context,
                        options=o or options,
                        question=q or question,
                        skip_value_in_inputs=bool(q is not None),
                    )
                    assistant_continuation = None

                # Get grammar
                if enable_constrained_decoding and grammar:
                    if grammar_is_collection:
                        # Different grammars for each item
                        grammar_str = grammar[idx](v)
                    else:
                        grammar_str = grammar(v)
                else:
                    grammar_str = None

                yield GenerationItem(
                    identifier=curr_identifier,
                    prompt=item_prompt,
                    assistant_continuation=assistant_continuation,
                    grammar=grammar_str,
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
                len(active_tasks) < n_parallel * 3
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
        try:
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
                        result.value.split(grammar_suffix)[0]
                        if grammar_suffix
                        else result.value,
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
        finally:
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
