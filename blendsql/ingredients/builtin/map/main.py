import logging
import os
import typing as t
from pathlib import Path
import json
import pandas as pd
from colorama import Fore
from attr import attrs, attrib
import copy
from tqdm.auto import tqdm

from blendsql.configure import add_to_global_history
from blendsql.common.logger import logger
from blendsql.common.constants import DEFAULT_ANS_SEP, INDENT, DEFAULT_CONTEXT_FORMATTER
from blendsql.models import Model, ConstrainedModel
from blendsql.models.utils import user
from blendsql.models.constrained.utils import LMString, maybe_load_lm
from blendsql.ingredients.ingredient import MapIngredient
from blendsql.common.exceptions import IngredientException
from blendsql.ingredients.utils import (
    initialize_retriever,
    partialclass,
)
from blendsql.configure import MAX_OPTIONS_IN_PROMPT_KEY, DEFAULT_MAX_OPTIONS_IN_PROMPT
from blendsql.types import DataType, prepare_datatype
from .examples import (
    MapExample,
    AnnotatedMapExample,
    ConstrainedMapExample,
    ConstrainedAnnotatedMapExample,
    UnconstrainedMapExample,
    UnconstrainedAnnotatedMapExample,
    ContextType,
)
from blendsql.search.searcher import Searcher

DEFAULT_MAP_FEW_SHOT: t.List[AnnotatedMapExample] = [
    AnnotatedMapExample(**d)
    for d in json.loads(
        open(Path(__file__).resolve().parent / "./default_examples.json", "r").read()
    )
]
CONSTRAINED_MAIN_INSTRUCTION = "Complete the docstring for the provided Python function. The output should correctly answer the question provided for each input value. "
CONSTRAINED_MAIN_INSTRUCTION = (
    CONSTRAINED_MAIN_INSTRUCTION
    + "On each newline, you will follow the format of f({value}) == {answer}.\n"
)
DEFAULT_CONSTRAINED_MAP_BATCH_SIZE = 1

UNCONSTRAINED_MAIN_INSTRUCTION = (
    "Given a set of values from a database, answer the question for each value. "
)
UNCONSTRAINED_MAIN_INSTRUCTION = (
    UNCONSTRAINED_MAIN_INSTRUCTION
    + f" Your output should be separated by '{DEFAULT_ANS_SEP}', answering for each of the values left-to-right.\n"
)
DEFAULT_UNCONSTRAINED_MAP_BATCH_SIZE = 5


@attrs
class LLMMap(MapIngredient):
    DESCRIPTION = """
    If question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column using the scalar function:
        `{{LLMMap('question', 'table::column')}}`
    """
    model: Model = attrib(default=None)
    few_shot_retriever: t.Callable[[str], t.List[AnnotatedMapExample]] = attrib(
        default=None
    )
    list_options_in_prompt: bool = attrib(default=True)
    few_shot_retriever: t.Callable[[str], t.List[AnnotatedMapExample]] = attrib(
        default=None
    )
    context_formatter: t.Callable[[pd.DataFrame], str] = attrib(
        default=DEFAULT_CONTEXT_FORMATTER,
    )
    batch_size: int = attrib(default=None)

    @classmethod
    def from_args(
        cls,
        model: t.Optional[Model] = None,
        few_shot_examples: t.Optional[
            t.Union[t.List[dict], t.List[AnnotatedMapExample]]
        ] = None,
        list_options_in_prompt: bool = True,
        batch_size: t.Optional[int] = None,
        num_few_shot_examples: t.Optional[int] = None,
        searcher: t.Optional[Searcher] = None,
        enable_constrained_decoding: bool = True,
    ):
        """Creates a partial class with predefined arguments.

        Args:
            model: The model to be used. Defaults to None.
            few_shot_examples: A list of dictionary MapExample few-shot examples.
               If not specified, will use [default_examples.json](https://github.com/parkervg/blendsql/blob/main/blendsql/ingredients/builtin/map/default_examples.json) as default.
            list_options_in_prompt: Whether to list options in the prompt. Defaults to True.
            batch_size: The batch size for processing. Defaults to 5.
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
        else:
            # Sort of guessing here - the user could change the `model` type later,
            #   or pass the model at the `BlendSQL(...)` level instead of the ingredient level.
            if model is not None:
                few_shot_examples = [
                    ConstrainedAnnotatedMapExample(**d)
                    if isinstance(d, dict)
                    else ConstrainedAnnotatedMapExample(**d.__dict__)
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
                list_options_in_prompt=list_options_in_prompt,
                batch_size=batch_size,
                searcher=searcher,
                enable_constrained_decoding=enable_constrained_decoding,
            )
        )

    def run(
        self,
        model: Model,
        question: str,
        values: t.List[str],
        context_formatter: t.Callable[[pd.DataFrame], str],
        list_options_in_prompt: bool,
        unpacked_questions: t.List[str] = None,
        searcher: t.Optional[Searcher] = None,
        options: t.Optional[t.List[str]] = None,
        few_shot_retriever: t.Callable[
            [str], t.List[ConstrainedAnnotatedMapExample]
        ] = None,
        value_limit: t.Optional[int] = None,
        example_outputs: t.Optional[str] = None,
        return_type: t.Optional[t.Union[DataType, str]] = None,
        regex: t.Optional[str] = None,
        context: t.Optional[pd.DataFrame] = None,
        batch_size: int = None,
        **kwargs,
    ) -> t.List[t.Union[float, int, str, bool]]:
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
            raise IngredientException(
                "LLMMap requires a `Model` object, but nothing was passed!\nMost likely you forgot to set the `default_model` argument in `blend()`"
            )
        if few_shot_retriever is None:
            few_shot_retriever = lambda *_: DEFAULT_MAP_FEW_SHOT

        context_in_use: t.List[str] = [None] * len(values)
        context_in_use_type: ContextType = None
        # If we explicitly passed `context`, this should take precedence over the vector store.
        if searcher is not None and context is None:
            if unpacked_questions:  # Implies we have different context for each value
                context_in_use = searcher(unpacked_questions)
                logger.debug(
                    Fore.LIGHTBLACK_EX
                    + f"Retrieved contexts '{[str(d[:2]) + '...' for d in context_in_use[:3]]}...'"
                    + Fore.RESET
                )
                context_in_use_type = ContextType.LOCAL
            else:
                context_in_use = " | ".join(searcher(question)[0])
                logger.debug(
                    Fore.LIGHTBLACK_EX
                    + f"Retrieved context '{context_in_use[:50]}...'"
                    + Fore.RESET
                )
                context_in_use_type = ContextType.GLOBAL
        elif context is not None:  # If we've passed a table context
            if isinstance(context, pd.DataFrame):
                context_in_use = context_formatter(context)
            context_in_use_type = ContextType.GLOBAL

        # Unpack default kwargs
        table_name, column_name = self.unpack_default_kwargs(**kwargs)
        if value_limit is not None:
            values = values[:value_limit]
        values = [value if not pd.isna(value) else "-" for value in values]
        resolved_return_type: DataType = prepare_datatype(
            return_type=return_type, options=options, quantifier=None
        )
        current_example = MapExample(
            question=question,
            column_name=column_name,
            table_name=table_name,
            return_type=resolved_return_type,
            example_outputs=example_outputs,
            options=options,
            context_type=context_in_use_type,
            context=context_in_use,
        )
        if isinstance(model, ConstrainedModel):
            batch_size = batch_size or DEFAULT_CONSTRAINED_MAP_BATCH_SIZE
            current_example = ConstrainedMapExample(**current_example.__dict__)
            few_shot_examples: t.List[ConstrainedAnnotatedMapExample] = [
                ConstrainedAnnotatedMapExample(**example.__dict__)
                for example in few_shot_retriever(current_example.to_string())
            ]
        else:
            batch_size = batch_size or DEFAULT_UNCONSTRAINED_MAP_BATCH_SIZE
            current_example = UnconstrainedMapExample(**current_example.__dict__)
            few_shot_examples: t.List[UnconstrainedAnnotatedMapExample] = [
                UnconstrainedAnnotatedMapExample(**example.__dict__)
                for example in few_shot_retriever(
                    current_example.to_string(values=values)
                )
            ]

        regex = regex or resolved_return_type.regex

        if options is not None and list_options_in_prompt:
            if len(options) > int(
                os.getenv(MAX_OPTIONS_IN_PROMPT_KEY, DEFAULT_MAX_OPTIONS_IN_PROMPT)
            ):
                logger.debug(
                    Fore.YELLOW
                    + f"Number of options ({len(options)}) is greater than the configured MAX_OPTIONS_IN_PROMPT.\nWill run inference without explicitly listing these options in the prompt text."
                )
                list_options_in_prompt = False

        if isinstance(model, ConstrainedModel):
            import guidance

            if all(x is not None for x in [options, regex]):
                raise IngredientException(
                    "MapIngredient exception!\nCan't have both `options` and `regex` argument passed."
                )

            lm = LMString()  # type: ignore

            if options is not None and self.enable_constrained_decoding:
                gen_f = lambda _: guidance.select(options=options)  # type: ignore
            elif (
                resolved_return_type.name == "substring"
                and self.enable_constrained_decoding
            ):
                # Special case for substring datatypes
                gen_f = lambda s: guidance.substring(target_string=s)
            else:
                if not self.enable_constrained_decoding:
                    logger.debug(
                        Fore.YELLOW
                        + "Not applying constraints, since `enable_constrained_decoding==False`"
                        + Fore.RESET
                    )
                gen_f = lambda _: guidance.gen(
                    max_tokens=kwargs.get("max_tokens", 200),
                    # guidance=0.2.1 doesn't allow both `stop` and `regex` to be passed
                    stop=None
                    if regex is not None
                    else [")", f"\n{INDENT()}"]
                    + (['"'] if resolved_return_type.name == "str" else []),
                    regex=regex,
                )  # type: ignore

            def make_prediction(
                value: str,
                context: t.Optional[t.Union[str, t.List[str]]],
                str_output: bool,
                gen_f: t.Callable,
            ) -> str:
                """If `context` is a string, it is a serialized table subset.
                Else, it's a list of documents.
                """

                def get_quote(s: str):
                    return '"""' if any(c in s for c in ["\n", '"']) else '"'

                value_quote = get_quote(value)
                if isinstance(
                    context, list
                ):  # If it's a string, it's already been added in docstring as global context
                    gen_str = f"""{INDENT(2)}f(\n{INDENT(3)}{value_quote}{value}{value_quote}"""
                    json_str = json.dumps(context, ensure_ascii=False, indent=20)[:-1]
                    gen_str += (
                        f", \n{INDENT(3)}" + json_str + f"{INDENT(3)}]\n{INDENT(2)})"
                    )
                else:
                    indented_value = value.replace("\n", f"\n{INDENT(2)}")
                    gen_str = (
                        f"""{INDENT(2)}f({value_quote}{indented_value}{value_quote})"""
                    )
                gen_str += f""" == {'"' if str_output else ''}{guidance.capture(gen_f(value), name=value)}{'"' if str_output else ''}"""
                return gen_str

            example_str = ""
            if len(few_shot_examples) > 0:
                for example in few_shot_examples:
                    example_str += example.to_string()

            loaded_lm = False
            batch_inference_strings = []
            batch_inference_values = []
            value_to_cache_key = {}
            for c, v in zip(context_in_use, values):
                if context_in_use_type == ContextType.LOCAL:
                    current_example.context = c

                current_example_str = current_example.to_string(
                    list_options=list_options_in_prompt,
                    add_leading_newlines=True,
                )

                # First check - do we need to load the model?
                in_cache = False
                if model.caching:
                    cached_response, cache_key = model.check_cache(
                        CONSTRAINED_MAIN_INSTRUCTION,
                        example_str,
                        current_example_str,
                        question,
                        options,
                        c,
                        v,
                        funcs=[make_prediction, gen_f],
                    )
                    if cached_response is not None:
                        lm = lm.set(v, cached_response)
                        in_cache = True
                    else:
                        value_to_cache_key[v] = cache_key

                if not in_cache and not loaded_lm:
                    lm: guidance.models.Model = maybe_load_lm(model, lm)
                    loaded_lm = True
                    with guidance.user():
                        lm += CONSTRAINED_MAIN_INSTRUCTION
                        lm += example_str
                        lm += current_example_str

                model.prompt_tokens += len(model.tokenizer.encode(current_example_str))

                if not in_cache:
                    batch_inference_strings.append(
                        make_prediction(
                            value=v,
                            context=c,
                            str_output=(resolved_return_type.name == "str"),
                            gen_f=gen_f,
                        )
                    )
                    batch_inference_values.append(v)

            with guidance.assistant():
                iter = range(0, len(batch_inference_strings), batch_size)
                if logger.level <= logging.DEBUG:
                    # Wrap with tqdm if `verbose=True`
                    iter = tqdm(
                        iter,
                        total=len(batch_inference_strings) // batch_size,
                        desc=f"LLMMap with batch_size={batch_size}",
                    )
                for i in iter:
                    model.num_generation_calls += 1
                    batch_lm = lm + "\n".join(
                        batch_inference_strings[i : i + batch_size]
                    )
                    lm._interpreter.state.captures.update(
                        batch_lm._interpreter.state.captures
                    )
                    add_to_global_history(str(batch_lm))
                    if model.caching:
                        for value in batch_inference_values[i : i + batch_size]:
                            cache_key = value_to_cache_key[value]
                            model.cache[cache_key] = lm.get(value)  # type: ignore

            lm_mapping: t.List[str] = [lm[value] for value in values]  # type: ignore
            model.completion_tokens += sum(
                [len(model.tokenizer.encode(v)) for v in lm_mapping]
            )
            model.prompt_tokens += lm._get_usage().input_tokens
            # For each value, call the DataType's `coerce_fn()`
            mapped_values = [resolved_return_type.coerce_fn(s) for s in lm_mapping]
        else:
            sorted_values = sorted(values)
            messages_list: t.List[t.List[dict]] = []
            batch_sizes: t.List[int] = []
            if current_example.context_type == ContextType.LOCAL:
                logger.debug(
                    Fore.YELLOW
                    + f"Overriding batch_size={batch_size} to 0, since UnconstrainedModels with LLMMap don't support local context for now"
                    + Fore.RESET
                )
                batch_size = 1
                current_example.context_type = ContextType.GLOBAL
                current_example.context = None
            for i in range(0, len(sorted_values), batch_size):
                curr_batch_values = sorted_values[i : i + batch_size]
                curr_batch_contexts = context_in_use[i : i + batch_size]
                batch_sizes.append(len(curr_batch_values))
                current_batch_example = copy.deepcopy(current_example)
                user_msg_str = ""
                user_msg_str += UNCONSTRAINED_MAIN_INSTRUCTION
                # Add few-shot examples
                for example in few_shot_examples:
                    user_msg_str += example.to_string()
                # Add the current question + context for inference
                if current_batch_example.context_type == ContextType.GLOBAL:
                    current_batch_example.context = "\n".join(curr_batch_contexts[0])
                user_msg_str += current_batch_example.to_string(
                    values=curr_batch_values,
                    list_options=list_options_in_prompt,
                )
                messages_list.append([user(user_msg_str)])
            add_to_global_history(messages_list)
            responses: t.List[str] = model.generate(
                messages_list=messages_list, max_tokens=kwargs.get("max_tokens", None)
            )

            # Post-process language model response
            mapped_values = []
            total_missing_values = 0
            for idx, r in enumerate(responses):
                expected_len = batch_sizes[idx]
                predictions: t.List[Union[str, None]] = r.split(DEFAULT_ANS_SEP)  # type: ignore
                while len(predictions) < expected_len:
                    total_missing_values += 1
                    predictions.append(None)
                # Try to map to booleans, `None`, and numeric datatypes
                mapped_values.extend(
                    [current_example.return_type.coerce_fn(s) for s in predictions]
                )

            mapping = {k: v for k, v in zip(sorted_values, mapped_values)}
            mapped_values = [mapping[value] for value in values]

            if total_missing_values > 0:
                logger.debug(
                    Fore.RED
                    + f"LLMMap with {type(model).__name__}({model.model_name_or_path}) only returned {len(mapped_values) - total_missing_values} out of {len(mapped_values)} values"
                )

        logger.debug(
            Fore.YELLOW
            + f"Finished LLMMap with values:\n{json.dumps(dict(zip(values[:10], mapped_values[:10])), indent=4)}"
            + Fore.RESET
        )
        if os.getenv("BLENDSQL_ALWAYS_LOWERCASE_RESPONSE") == "1":
            # Basic transforms not handled by SQLite type affinity
            return [{"True": True, "False": False}.get(v, v) for v in mapped_values]
        return mapped_values
