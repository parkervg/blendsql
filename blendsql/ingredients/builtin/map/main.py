import copy
import os
import typing as t
from collections.abc import Collection
from pathlib import Path
import json
import pandas as pd
from colorama import Fore
from attr import attrs, attrib

from blendsql.common.logger import logger
from blendsql.models import Model, ConstrainedModel
from blendsql.models.constrained.utils import LMString, maybe_load_lm
from blendsql.models.utils import user
from blendsql.common import constants as CONST
from blendsql.ingredients.ingredient import MapIngredient
from blendsql.common.exceptions import IngredientException
from blendsql.ingredients.utils import (
    initialize_retriever,
    partialclass,
)
from blendsql.configure import MAX_OPTIONS_IN_PROMPT_KEY, DEFAULT_MAX_OPTIONS_IN_PROMPT
from blendsql.types import DataType, prepare_datatype
from .examples import (
    AnnotatedMapExample,
    ConstrainedMapExample,
    ConstrainedAnnotatedMapExample,
    UnconstrainedAnnotatedMapExample,
    UnconstrainedMapExample,
)
from blendsql.search.faiss_vector_store import FaissVectorStore

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
DEFAULT_CONSTRAINED_MAP_BATCH_SIZE = 100

UNCONSTRAINED_MAIN_INSTRUCTION = (
    "Given a set of values from a database, answer the question for each value. "
)
UNCONSTRAINED_MAIN_INSTRUCTION = (
    UNCONSTRAINED_MAIN_INSTRUCTION
    + " Your output should be separated by ';', answering for each of the values left-to-right.\n"
)
DEFAULT_UNCONSTRAINED_MAP_BATCH_SIZE = 5

OPTIONS_INSTRUCTION = "Your responses MUST select from one of the following values:\n"


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
        default=lambda df: json.dumps(df.to_dict(orient="records"), indent=4),
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
        k: t.Optional[int] = None,
        vector_store: t.Optional[FaissVectorStore] = None,
    ):
        """Creates a partial class with predefined arguments.

        Args:
            model: The model to be used. Defaults to None.
            few_shot_examples: A list of dictionary MapExample few-shot examples.
               If not specified, will use [default_examples.json](https://github.com/parkervg/blendsql/blob/main/blendsql/ingredients/builtin/map/default_examples.json) as default.
            list_options_in_prompt: Whether to list options in the prompt. Defaults to True.
            batch_size: The batch size for processing. Defaults to 5.
            k: Determines number of few-shot examples to use for each ingredient call.
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
                    # Will fetch `k` most relevant few-shot examples using embedding-based retriever
                    k=2,
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
            if model is not None and isinstance(model, ConstrainedModel):
                few_shot_examples = [
                    ConstrainedAnnotatedMapExample(**d)
                    if isinstance(d, dict)
                    else ConstrainedAnnotatedMapExample(**d.__dict__)
                    for d in few_shot_examples
                ]
            else:
                few_shot_examples = [
                    UnconstrainedAnnotatedMapExample(**d)
                    if isinstance(d, dict)
                    else UnconstrainedAnnotatedMapExample(**d.__dict__)
                    for d in few_shot_examples
                ]
        few_shot_retriever = initialize_retriever(examples=few_shot_examples, k=k)
        return cls._maybe_set_name_to_var_name(
            partialclass(
                cls,
                model=model,
                few_shot_retriever=few_shot_retriever,
                list_options_in_prompt=list_options_in_prompt,
                batch_size=batch_size,
                vector_store=vector_store,
            )
        )

    def run(
        self,
        model: Model,
        question: str,
        values: t.List[str],
        context_formatter: t.Callable[[pd.DataFrame], str],
        list_options_in_prompt: bool,
        vector_store: t.Optional[FaissVectorStore] = None,
        options: t.Optional[Collection[str]] = None,
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
        # if few_shot_retriever is None:
        few_shot_retriever = lambda *_: DEFAULT_MAP_FEW_SHOT
        use_context = context is not None or vector_store is not None
        # If we explicitly passed `context`, this should take precedence over the vector store.
        if vector_store is not None and context is None:
            # Concatenate each value to the front of the questions
            # E.g. 'What year were they born?' -> 'Ryan Lochte What year were they born?'
            docs = vector_store([f"{v} {question}" for v in values], values=values)
            context = ["\n\n".join(d) for d in docs]
            logger.debug(
                Fore.LIGHTBLACK_EX
                + f"Retrieved contexts '{[d[:50] + '...' for d in context]}'"
                + Fore.RESET
            )
        elif context is None:
            context = [None] * len(values)

        # Unpack default kwargs
        table_name, column_name = self.unpack_default_kwargs(**kwargs)
        if value_limit is not None:
            values = values[:value_limit]
        values = [value if not pd.isna(value) else "-" for value in values]
        resolved_return_type: DataType = prepare_datatype(
            return_type=return_type, options=options, quantifier=None
        )

        if isinstance(model, ConstrainedModel):
            current_example = ConstrainedMapExample(
                question=question,
                column_name=column_name,
                table_name=table_name,
                return_type=resolved_return_type,
                example_outputs=example_outputs,
                options=options,
                use_context=use_context,
            )
        else:
            current_example = UnconstrainedMapExample(
                question=question,
                column_name=column_name,
                table_name=table_name,
                return_type=resolved_return_type,
                example_outputs=example_outputs,
                options=options,
                use_context=use_context,
                values=values,
            )

        regex = regex or resolved_return_type.regex

        if isinstance(model, ConstrainedModel):
            batch_size = batch_size or DEFAULT_CONSTRAINED_MAP_BATCH_SIZE
            few_shot_examples: t.List[ConstrainedAnnotatedMapExample] = [
                ConstrainedAnnotatedMapExample(**example.__dict__)
                for example in few_shot_retriever()
            ]
        else:
            batch_size = batch_size or DEFAULT_UNCONSTRAINED_MAP_BATCH_SIZE
            few_shot_examples: t.List[UnconstrainedAnnotatedMapExample] = [
                UnconstrainedAnnotatedMapExample(**example.__dict__)
                for example in few_shot_retriever()
            ]

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

            if options is not None:
                gen_f = lambda _: guidance.select(options=options)  # type: ignore
            elif resolved_return_type.name == "substring":
                # Special case for substring datatypes
                gen_f = lambda s: guidance.substring(target_string=s)
            else:
                gen_f = lambda _: guidance.gen(
                    max_tokens=kwargs.get("max_tokens", 200),
                    # guidance=0.2.1 doesn't allow both `stop` and `regex` to be passed
                    stop=["\n\t", ")"]
                    + (['"'] if current_example.return_type.name == "str" else []),
                    regex=regex,
                )  # type: ignore

            def make_prediction(
                value: str,
                context: t.Optional[str],
                str_output: bool,
                gen_f: t.Callable,
            ) -> str:
                def get_quote(s: str):
                    return '"""' if any(c in s for c in ["\n", '"']) else '"'

                value_quote = get_quote(value)
                gen_str = f"""\t\tf({value_quote}{value}{value_quote}"""
                if context is not None:
                    context_quote = get_quote(context)
                    gen_str += f""", {context_quote}{context}{context_quote}"""
                gen_str += f""") == {'"' if str_output else ''}{guidance.capture(gen_f(value), name=value)}{'"' if str_output else ''}"""
                return gen_str

            example_str = ""
            if len(few_shot_examples) > 0:
                for example in few_shot_examples:
                    example_str += example.to_string()

            loaded_lm = False
            batch_inference_strings = []
            cache_keys = []
            # Due to guidance's prefix caching, this is a one-time cost
            model.prompt_tokens += len(
                model.tokenizer.encode(CONSTRAINED_MAIN_INSTRUCTION + example_str)
            )
            # for i in range(0, len(sorted_values), batch_size):
            for c, v in zip(context, values):
                current_example.context = c
                current_example_str = current_example.to_string(
                    list_options=list_options_in_prompt,
                    add_leading_newlines=True,
                )

                # First check - do we need to load the model?
                in_cache = False
                if model.caching:
                    responses, key = model.check_cache(
                        CONSTRAINED_MAIN_INSTRUCTION,
                        example_str,
                        current_example_str,
                        question,
                        c,
                        v,
                        funcs=[make_prediction, gen_f],
                    )
                    if responses is not None:
                        lm._variables.update(responses)
                        in_cache = True
                        cache_keys.append(None)
                    else:
                        cache_keys.append(key)
                if not in_cache and not loaded_lm:
                    lm: guidance.models.Model = maybe_load_lm(model, lm)
                    loaded_lm = True
                    with guidance.user():
                        lm += CONSTRAINED_MAIN_INSTRUCTION
                        lm += example_str
                        lm += current_example_str

                model.prompt_tokens += len(model.tokenizer.encode(current_example_str))

                if not in_cache:
                    model.num_generation_calls += 1
                    batch_inference_strings.append(
                        make_prediction(
                            value=v,
                            context=c,
                            str_output=(resolved_return_type.name == "str"),
                            gen_f=gen_f,
                        )
                    )

            with guidance.assistant():
                for i in range(0, len(batch_inference_strings), batch_size):
                    batch_lm = lm + "\n".join(
                        batch_inference_strings[i : i + batch_size]
                    )
                    generated_batch_variables = {
                        v: batch_lm.get(v) for v in values[i : i + batch_size]
                    }
                    lm._variables.update(generated_batch_variables)
            if model.caching:
                for cache_key, value in zip(cache_keys, values):
                    if cache_key is None:
                        continue
                    model.cache[cache_key] = {value: lm.get(value)}  # type: ignore

            lm_mapping: t.List[str] = [lm[value] for value in values]  # type: ignore
            model.completion_tokens += sum(
                [len(model.tokenizer.encode(v)) for v in lm_mapping]
            )
            # For each value, call the DataType's `coerce_fn()`
            mapped_values = [resolved_return_type.coerce_fn(s) for s in lm_mapping]
        else:
            sorted_values = sorted(values)
            messages_list: t.List[t.List[dict]] = []
            batch_sizes: t.List[int] = []
            for i in range(0, len(sorted_values), batch_size):
                curr_batch_values = sorted_values[i : i + batch_size]
                batch_sizes.append(len(curr_batch_values))
                current_batch_example = copy.deepcopy(current_example)
                current_batch_example.values = curr_batch_values
                user_msg_str = ""
                user_msg_str += UNCONSTRAINED_MAIN_INSTRUCTION
                # Add few-shot examples
                for example in few_shot_examples:
                    user_msg_str += example.to_string(include_values=False)
                # Add the current question + context for inference
                user_msg_str += current_batch_example.to_string(
                    list_options=list_options_in_prompt, include_values=True
                )
                messages_list.append([user(user_msg_str)])

            responses: t.List[str] = model.generate(
                messages_list=messages_list, max_tokens=kwargs.get("max_tokens", None)
            )

            # Post-process language model response
            mapped_values = []
            total_missing_values = 0
            for idx, r in enumerate(responses):
                expected_len = batch_sizes[idx]
                predictions: t.List[Union[str, None]] = r.split(CONST.DEFAULT_ANS_SEP)  # type: ignore
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
        return mapped_values
