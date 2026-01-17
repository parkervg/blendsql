import ast
import logging
import os
from typing import Callable
from pathlib import Path
import json
import polars as pl
import pandas as pd
from dataclasses import dataclass, field
import copy
from itertools import islice, repeat
from tqdm.auto import tqdm
from textwrap import indent, dedent

from blendsql.configure import add_to_global_history
from blendsql.common.logger import logger, Color
from blendsql.common.constants import DEFAULT_ANS_SEP, INDENT, DEFAULT_CONTEXT_FORMATTER
from blendsql.models import Model, ConstrainedModel
from blendsql.models.utils import user
from blendsql.models.constrained.utils import LMString, maybe_load_lm
from blendsql.ingredients.ingredient import MapIngredient
from blendsql.common.exceptions import LMFunctionException
from blendsql.common.typing import DataType, QuantifierType, AdditionalMapArg
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
)
from blendsql.types import prepare_datatype, apply_type_conversion, unquote
from .examples import (
    MapExample,
    AnnotatedMapExample,
    ConstrainedMapExample,
    ConstrainedAnnotatedMapExample,
    UnconstrainedMapExample,
    UnconstrainedAnnotatedMapExample,
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
DEFAULT_CONSTRAINED_MAP_BATCH_SIZE = 3

UNCONSTRAINED_MAIN_INSTRUCTION = (
    "Given a set of values from a database, answer the question for each value. "
)
UNCONSTRAINED_MAIN_INSTRUCTION = (
    UNCONSTRAINED_MAIN_INSTRUCTION
    + f" Your output should be separated by '{DEFAULT_ANS_SEP}', answering for each of the values left-to-right.\n"
)
DEFAULT_UNCONSTRAINED_MAP_BATCH_SIZE = 5


@dataclass
class LLMMap(MapIngredient):
    DESCRIPTION = """
    If question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column using the scalar function:
        `{{LLMMap('question', 'table::column')}}`
    """
    model: Model = field(default=None)
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
        model: Model | None = None,
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

    def run(
        self,
        model: Model,
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
        few_shot_retriever: Callable[
            [str], list[ConstrainedAnnotatedMapExample]
        ] = None,
        value_limit: int | None = None,
        example_outputs: str | None = None,
        quantifier: QuantifierType = None,
        return_type: DataType | str | None = None,
        regex: str | None = None,
        batch_size: int = None,
        exit_condition: Callable = None,
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
            if not isinstance(model, ConstrainedModel):
                raise NotImplementedError(
                    "`options_searcher` logic not yet implemented for `UnconstrainedModel` classes.\nUse a `ConstrainedModel` class for now (`LlamaCpp` or `TransformersLLM`)."
                )
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

        if isinstance(model, ConstrainedModel):
            batch_size = batch_size or DEFAULT_CONSTRAINED_MAP_BATCH_SIZE
            current_example = ConstrainedMapExample(**current_example.__dict__)
            few_shot_examples: list[ConstrainedAnnotatedMapExample] = [
                ConstrainedAnnotatedMapExample(**example.__dict__)
                if not isinstance(example, dict)
                else ConstrainedAnnotatedMapExample(**example)
                for example in few_shot_retriever(
                    current_example.to_string(
                        additional_args=additional_args,
                    )
                )
            ]
        else:
            batch_size = batch_size or DEFAULT_UNCONSTRAINED_MAP_BATCH_SIZE
            current_example = UnconstrainedMapExample(**current_example.__dict__)
            few_shot_examples: list[UnconstrainedAnnotatedMapExample] = [
                UnconstrainedAnnotatedMapExample(**example.__dict__)
                if not isinstance(example, dict)
                else UnconstrainedMapExample(**example)
                for example in few_shot_retriever(
                    current_example.to_string(values=values)
                )
            ]

        is_list_output = resolved_return_type.quantifier is not None
        regex = regex or resolved_return_type.regex
        quantifier = resolved_return_type.quantifier

        if isinstance(model, ConstrainedModel):
            import guidance

            lm = LMString()  # type: ignore

            if all(x is not None for x in [options, regex]):
                raise LMFunctionException(
                    "MapIngredient exception!\nCan't have both `options` and `regex` argument passed."
                )

            gen_f = None
            if enable_constrained_decoding:
                if is_list_output:
                    if self.options_searcher is not None:
                        # Need to create separate gen_f for each set of filtered_options
                        gen_f = [
                            lambda _, o=o: gen_list(
                                force_quotes=resolved_return_type.requires_quotes,
                                quantifier=quantifier,
                                options=o,
                                regex=regex,
                            )
                            for o in filtered_options
                        ]
                    else:
                        gen_f = lambda _: gen_list(
                            force_quotes=resolved_return_type.requires_quotes,
                            quantifier=quantifier,
                            options=options,
                            regex=regex,
                        )
                else:
                    if self.options_searcher is not None:
                        # Need to create separate gen_f for each set of filtered_options
                        gen_f = [
                            lambda _, o=o: _wrap_with_quotes(
                                guidance.select(options=o),
                                has_options_or_regex=bool(o or regex),
                                force_quotes=resolved_return_type.requires_quotes,
                            )
                            for o in filtered_options
                        ]
                    elif options is not None:
                        select_fn = guidance.select(options=options)
                        gen_f = lambda _: _wrap_with_quotes(
                            select_fn,
                            has_options_or_regex=bool(options or regex),
                            force_quotes=resolved_return_type.requires_quotes,
                        )
                    elif resolved_return_type.name == "substring":
                        # Special case for substring datatypes
                        gen_f = lambda s: _wrap_with_quotes(
                            guidance.substring(target_string=s),
                            has_options_or_regex=bool(options or regex),
                            force_quotes=resolved_return_type.requires_quotes,
                        )
            else:
                logger.debug(
                    Color.warning(
                        "Not applying constraints, since `enable_constrained_decoding==False`"
                    )
                )
            if gen_f is None:
                # Create base gen_f function
                gen_f = lambda _: _wrap_with_quotes(
                    guidance.gen(
                        max_tokens=kwargs.get(
                            "max_tokens",
                            int(os.getenv(MAX_TOKENS_KEY, DEFAULT_MAX_TOKENS)),
                        ),
                        # guidance>=0.2.1 doesn't allow both `stop` and `regex` to be passed
                        stop=None
                        if regex is not None
                        else [")", f"\n{INDENT()}"]
                        + (['"'] if resolved_return_type.requires_quotes else []),
                        regex=regex if enable_constrained_decoding else None,
                    ),
                    has_options_or_regex=bool(options or regex),
                    force_quotes=resolved_return_type.requires_quotes,
                )

            def make_prediction(
                identifier: str,
                value: str,
                additional_args: list[AdditionalMapArg] | None,
                context: str | list[str] | None,
                context_in_use_type: FeatureType | None,
                local_options: list[str] | None,
                gen_f: Callable,
            ) -> str:
                def get_quote(s: str):
                    return '"""' if any(c in s for c in ["\n", '"']) else '"'

                value_quote = get_quote(value)
                has_more_than_one_arg = bool(
                    context_in_use_type == FeatureType.LOCAL
                    or additional_args is not None
                    or local_options is not None
                )
                if has_more_than_one_arg:
                    # If we pass more than one arg, make them appear on newlines
                    gen_str = f"""{INDENT(2)}f(\n{INDENT(3)}{value_quote}{value}{value_quote}"""
                    if additional_args is not None:
                        for arg in additional_args:
                            gen_str += f',\n{INDENT(3)}"{arg}"'
                    if context_in_use_type == FeatureType.LOCAL:
                        json_str = json.dumps(context, ensure_ascii=False, indent=16)[
                            :-1
                        ]
                        gen_str += f",\n{INDENT(3)}" + json_str + f"{INDENT(3)}]"
                    if local_options is not None:
                        gen_str += f",\n{INDENT(2)}{local_options}"
                else:  # Global contexts have already been handled. We only have a single variable to pass.
                    indented_value = value.replace("\n", f"\n{INDENT(2)}")
                    gen_str = (
                        f"""{INDENT(2)}f({value_quote}{indented_value}{value_quote}"""
                    )
                if has_more_than_one_arg:
                    gen_str += f"\n{INDENT(2)}"
                # Below, make sure we set the output to the `identifier` name
                # If we just did `name=value`, then this would lose the difference between
                #   identical values with different additional args / context
                gen_str += f""") == {guidance.capture(gen_f(value), name=identifier)}"""
                return gen_str

            example_str = ""
            if len(few_shot_examples) > 0:
                for example in few_shot_examples:
                    example_str += example.to_string()

            lm_mapping = {}  # Where we store the final type-cast results
            loaded_lm = False

            # Generator to yield prompts on-the-fly
            def generate_batch_items():
                """Generator that yields (identifier, prompt_string, cache_info) tuples"""
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
                    curr_identifier = v
                    if a is not None:
                        curr_identifier += f"_{a}"
                    if c is not None:
                        curr_identifier += f"_{c}"

                    # Check cache first
                    in_cache = False
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
                                make_prediction,
                                gen_f[idx]
                                if self.options_searcher is not None
                                and enable_constrained_decoding
                                else gen_f,
                            ],
                        )
                        if cached_response is not None:
                            lm_mapping[curr_identifier] = cached_response
                            in_cache = True

                    if in_cache:
                        continue  # Skip to next item

                    # Prepare the prompt only if not cached
                    if context_in_use_type == FeatureType.LOCAL:
                        current_example.context = c

                    current_example_str = current_example.to_string(
                        list_options=list_options_in_prompt,
                        additional_args=additional_args,
                        add_leading_newlines=True,
                    )

                    model.prompt_tokens += len(
                        model.tokenizer.encode(current_example_str)
                    )

                    prompt_string = make_prediction(
                        identifier=curr_identifier,
                        value=v,
                        additional_args=a,
                        context=c,
                        context_in_use_type=context_in_use_type,
                        local_options=o,
                        gen_f=gen_f[idx]
                        if self.options_searcher is not None
                        and enable_constrained_decoding
                        else gen_f,
                    )

                    yield (
                        curr_identifier,
                        current_example_str,
                        prompt_string,
                        cache_key,
                    )

            batch_generator = generate_batch_items()
            current_batch_identifiers = []
            current_batch_strings = []
            current_batch_cache_keys = []
            processed_items = 0
            total_items = len(values)
            all_processed_identifiers = []

            if logger.level <= logging.DEBUG:
                pbar = tqdm(
                    desc=(Color.prefix if Color.in_block else "")
                    + f"LLMMap with `{batch_size=}` and `{len(few_shot_examples)=}`",
                    total=total_items,
                )

            for (
                identifier,
                current_example_str,
                prompt_string,
                cache_key,
            ) in batch_generator:
                # Load LM on first non-cached item
                if not loaded_lm:
                    lm: guidance.models.Model = maybe_load_lm(model, lm)
                    loaded_lm = True
                    lm = model.maybe_add_system_prompt(lm)
                    with guidance.user():
                        lm += CONSTRAINED_MAIN_INSTRUCTION
                        if example_str != "":
                            lm += example_str
                            lm += "\n\nNow, complete the docstring for the following example:"
                        lm += current_example_str

                current_batch_identifiers.append(identifier)
                current_batch_strings.append(prompt_string)
                current_batch_cache_keys.append(cache_key)

                if len(current_batch_strings) >= batch_size:
                    with guidance.assistant():
                        batch_lm = lm + "\n".join(current_batch_strings)

                    # With guidance, each value is still getting its own generation call
                    # This logic is just wrapped up inside the single `batch_lm = lm  ...` call.
                    model.num_generation_calls += len(current_batch_strings)
                    self.num_values_passed += len(current_batch_strings)

                    model.completion_tokens += sum(
                        [
                            len(
                                model.tokenizer_encode(
                                    unquote(result_payload["value"])
                                    if resolved_return_type.requires_quotes
                                    else result_payload["value"]
                                )
                            )
                            for result_payload in batch_lm._interpreter.state.captures.values()
                        ]
                    )

                    batch_lm_mapping = {
                        value: apply_type_conversion(
                            result_payload["value"],
                            return_type=resolved_return_type,
                            db=self.db,
                        )
                        for value, result_payload in batch_lm._interpreter.state.captures.items()
                    }
                    lm_mapping.update(batch_lm_mapping)
                    add_to_global_history(str(batch_lm))

                    if model.caching:
                        for i, identifier in enumerate(current_batch_identifiers):
                            if current_batch_cache_keys[i] is not None:
                                model.cache[current_batch_cache_keys[i]] = lm_mapping[
                                    identifier
                                ]

                    processed_items += len(current_batch_strings)
                    if logger.level <= logging.DEBUG:
                        pbar.update(len(current_batch_strings))

                    all_processed_identifiers.extend(current_batch_identifiers)

                    # Clear batch
                    current_batch_identifiers = []
                    current_batch_strings = []
                    current_batch_cache_keys = []

                    # Check exit condition
                    if exit_condition is not None and exit_condition(lm_mapping):
                        logger.debug(
                            Color.optimization(
                                f"[ 🚪] Exit condition satisfied. Exiting early after processing {processed_items:,} out of {total_items:,} items."
                            )
                        )
                        break

            # Process any remaining items in the last partial batch
            if current_batch_strings and (
                exit_condition is None or not exit_condition(lm_mapping)
            ):
                model.num_generation_calls += 1
                with guidance.assistant():
                    batch_lm = lm + "\n".join(current_batch_strings)
                self.num_values_passed += len(current_batch_strings)

                model.completion_tokens += sum(
                    [
                        len(
                            model.tokenizer_encode(
                                unquote(result_payload["value"])
                                if resolved_return_type.requires_quotes
                                else result_payload["value"]
                            )
                        )
                        for result_payload in batch_lm._interpreter.state.captures.values()
                    ]
                )

                batch_lm_mapping = {
                    value: apply_type_conversion(
                        result_payload["value"],
                        return_type=resolved_return_type,
                        db=self.db,
                    )
                    for value, result_payload in batch_lm._interpreter.state.captures.items()
                }
                lm_mapping.update(batch_lm_mapping)
                add_to_global_history(str(batch_lm))

                if model.caching:
                    for i, identifier in enumerate(current_batch_identifiers):
                        if current_batch_cache_keys[i] is not None:
                            model.cache[current_batch_cache_keys[i]] = lm_mapping[
                                identifier
                            ]

                processed_items += len(current_batch_strings)
                if logger.level <= logging.DEBUG:
                    pbar.update(len(current_batch_strings))

                all_processed_identifiers.extend(current_batch_identifiers)

            if loaded_lm:
                model.prompt_tokens += lm._get_usage().input_tokens
                lm._reset_usage()

            mapped_values = [
                lm_mapping.get(identifier, None)
                for identifier in all_processed_identifiers
            ]
            # Find difference in length, and fill `None`
            mapped_values.extend(
                [None] * (len(values) - len(all_processed_identifiers))
            )

            logger.debug(
                lambda: Color.warning(
                    f"Finished LLMMap with {len(lm_mapping)} total values{' (10 shown)' if len(lm_mapping) > 10 else ''}:\n{indent(json.dumps({str(k): str(v) for k, v in islice(lm_mapping.items(), 10)}, indent=4), Color.prefix if Color.in_block else '')}"
                )
            )
            return mapped_values
        else:
            sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
            sorted_indices_to_original = {
                k: idx for idx, k in enumerate(sorted_indices)
            }
            sorted_values = [values[i] for i in sorted_indices]
            if context_in_use_type is not None:
                context = [context[i] for i in sorted_indices]

            messages_list: list[list[dict]] = []
            batch_sizes: list[int] = []
            if current_example.context_type == FeatureType.LOCAL:
                if batch_size != 1:
                    logger.debug(
                        Color.warning(
                            f"Overriding batch_size={batch_size} to 1, since UnconstrainedModels with LLMMap don't support local context for now"
                        )
                    )
                batch_size = 1
                current_example.context_type = FeatureType.GLOBAL
                current_example.context = None
            for i in range(0, len(sorted_values), batch_size):
                curr_batch_values = sorted_values[i : i + batch_size]
                curr_batch_contexts = context[i : i + batch_size]
                self.num_values_passed += len(curr_batch_values)
                batch_sizes.append(len(curr_batch_values))
                current_batch_example = copy.deepcopy(current_example)
                user_msg_str = ""
                user_msg_str += UNCONSTRAINED_MAIN_INSTRUCTION
                # Add few-shot examples
                for example in few_shot_examples:
                    user_msg_str += example.to_string()
                # Add the current question + context for inference
                if current_batch_example.context_type == FeatureType.GLOBAL:
                    str_context = "\n".join(curr_batch_contexts[0])
                    current_batch_example.context = str_context
                user_msg_str += current_batch_example.to_string(
                    values=curr_batch_values,
                    list_options=list_options_in_prompt,
                )
                messages_list.append([user(user_msg_str)])
            add_to_global_history(messages_list)
            responses: list[str] = model.generate(
                messages_list=messages_list,
                max_tokens=kwargs.get(
                    "max_tokens", int(os.getenv(MAX_TOKENS_KEY, DEFAULT_MAX_TOKENS))
                ),
            )
            model.num_generation_calls += len(messages_list)

            # Post-process language model response
            mapped_values = []
            total_missing_values = 0
            for idx, r in enumerate(responses):
                expected_len = batch_sizes[idx]
                predictions: list[t.Union[str, None]] = r.split(DEFAULT_ANS_SEP)  # type: ignore
                # Add null values, if we under-predicted
                while len(predictions) < expected_len:
                    total_missing_values += 1
                    predictions.append(None)
                # Cutoff, if we over-predicted
                predictions = predictions[:expected_len]
                if is_list_output:
                    curr_converted_preds = []
                    for pred in predictions:
                        try:
                            list_converted = ast.literal_eval(pred)
                        except (ValueError, SyntaxError):
                            logger.debug(
                                Color.error(
                                    f"Error casting prediction '{pred}' to a list"
                                )
                            )
                            curr_converted_preds.append([])
                            continue
                        for item in list_converted:
                            curr_converted_preds.append(
                                current_example.return_type.coerce_fn(item, self.db)
                            )
                    mapped_values.append(curr_converted_preds)
                else:
                    mapped_values.extend(
                        [
                            current_example.return_type.coerce_fn(s, self.db)
                            for s in predictions
                        ]
                    )
            mapped_values = [
                mapped_values[sorted_indices_to_original[i]]
                for i in range(len(mapped_values))
            ]

            if total_missing_values > 0:
                logger.debug(
                    Color.error(
                        f"LLMMap with {type(model).__name__}({model.model_name_or_path}) only returned {len(mapped_values) - total_missing_values} out of {len(mapped_values)} values"
                    )
                )
            logger.debug(
                lambda: Color.warning(
                    f"Finished LLMMap with {len(mapped_values)} values{' (10 shown)' if len(mapped_values) > 10 else ''}:\n{json.dumps(dict(islice(dict(zip(values, mapped_values)).items(), 10)), indent=4)}"
                )
            )
            return mapped_values
