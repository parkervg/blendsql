import copy
import os
from typing import Union, Optional, List, Callable
from collections.abc import Collection
from pathlib import Path
import json
import pandas as pd
from colorama import Fore
from attr import attrs, attrib

from blendsql._logger import logger
from blendsql.models import Model, ConstrainedModel
from blendsql.models.constrained.utils import LMString, maybe_load_lm
from blendsql.models._utils import user, assistant
from blendsql import _constants as CONST
from blendsql.ingredients.ingredient import MapIngredient
from blendsql._exceptions import IngredientException
from blendsql.ingredients.utils import (
    initialize_retriever,
    cast_responses_to_datatypes,
    prepare_datatype,
    partialclass,
)
from blendsql._configure import MAX_OPTIONS_IN_PROMPT_KEY, DEFAULT_MAX_OPTIONS_IN_PROMPT
from blendsql._constants import DataType
from .examples import AnnotatedMapExample, MapExample

DEFAULT_MAP_FEW_SHOT: List[AnnotatedMapExample] = [
    AnnotatedMapExample(**d)
    for d in json.loads(
        open(Path(__file__).resolve().parent / "./default_examples.json", "r").read()
    )
]
main_instruction = (
    "Given a set of values from a database, answer the question for each value. "
)
UNCONSTRAINED_MAIN_INSTRUCTION = (
    main_instruction
    + "Your output should be separated by ';', answering for each of the values left-to-right.\n"
)
CONSTRAINED_MAIN_INSTRUCTION = (
    main_instruction
    + "On each newline, you will follow the format of {value} -> {answer}.\n"
)
OPTIONS_INSTRUCTION = "Your responses MUST select from one of the following values:\n"
DEFAULT_MAP_BATCH_SIZE = 15


@attrs
class LLMMap(MapIngredient):
    DESCRIPTION = """
    If question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column using the scalar function:
        `{{LLMMap('question', 'table::column')}}`
    """
    model: Model = attrib(default=None)
    few_shot_retriever: Callable[[str], List[AnnotatedMapExample]] = attrib(
        default=None
    )
    list_options_in_prompt: bool = attrib(default=True)
    batch_size: int = attrib(default=DEFAULT_MAP_BATCH_SIZE)

    @classmethod
    def from_args(
        cls,
        model: Optional[Model] = None,
        few_shot_examples: Optional[
            Union[List[dict], List[AnnotatedMapExample]]
        ] = None,
        list_options_in_prompt: bool = True,
        batch_size: Optional[int] = DEFAULT_MAP_BATCH_SIZE,
        k: Optional[int] = None,
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
            few_shot_examples = [
                AnnotatedMapExample(**d) if isinstance(d, dict) else d
                for d in few_shot_examples
            ]
        few_shot_retriever = initialize_retriever(examples=few_shot_examples, k=k)
        return partialclass(
            cls,
            model=model,
            few_shot_retriever=few_shot_retriever,
            list_options_in_prompt=list_options_in_prompt,
            batch_size=batch_size,
        )

    def run(
        self,
        model: Model,
        question: str,
        values: List[str],
        list_options_in_prompt: bool,
        few_shot_retriever: Optional[Callable[[str], List[AnnotatedMapExample]]] = None,
        options: Optional[Collection[str]] = None,
        value_limit: Optional[int] = None,
        example_outputs: Optional[str] = None,
        output_type: Optional[Union[DataType, str]] = None,
        batch_size: int = DEFAULT_MAP_BATCH_SIZE,
        **kwargs,
    ) -> List[Union[float, int, str, bool]]:
        """For each value in a given column, calls a Model and retrieves the output.

        Args:
            question: The question to map onto the values. Will also be the new column name
            model: The Model (blender) we will make calls to.
            values: The list of values to apply question to.
            value_limit: Optional limit on the number of values to pass to the Model
            example_outputs: This gives the Model an example of the output we expect.
            output_type: In the absence of example_outputs, give the Model some signal as to what we expect as output.
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
        # Unpack default kwargs
        table_name, column_name = self.unpack_default_kwargs(**kwargs)
        if value_limit is not None:
            values = values[:value_limit]
        values = [value if not pd.isna(value) else "-" for value in values]
        output_type: DataType = prepare_datatype(
            output_type=output_type, options=options, modifier=None
        )
        current_example = MapExample(
            **{
                "question": question,
                "column_name": column_name,
                "table_name": table_name,
                "output_type": output_type,
                "example_outputs": example_outputs,
                "options": options,
                # Random subset of values for few-shot example retrieval
                # these will get replaced during batching later
                "values": values[:10],
            }
        )
        few_shot_examples: List[AnnotatedMapExample] = few_shot_retriever(
            current_example.to_string()
        )
        regex = None
        if current_example.output_type is not None:
            regex = current_example.output_type.regex
        options = current_example.options
        if options is not None and list_options_in_prompt:
            if len(options) > int(
                os.getenv(MAX_OPTIONS_IN_PROMPT_KEY, DEFAULT_MAX_OPTIONS_IN_PROMPT)
            ):
                logger.debug(
                    Fore.YELLOW
                    + f"Number of options ({len(options)}) is greater than the configured MAX_OPTIONS_IN_PROMPT.\nWill run inference without explicitly listing these options in the prompt text."
                )
                list_options_in_prompt = False
        sorted_values = sorted(values)  # Sort, to maximize cache hit rate
        if isinstance(model, ConstrainedModel):
            import guidance

            if all(x is not None for x in [options, regex]):
                raise IngredientException(
                    "MapIngredient exception!\nCan't have both `options` and `regex` argument passed."
                )

            lm = LMString()  # type: ignore

            if options is not None:
                gen_f = lambda: guidance.select(options=options)  # type: ignore
            elif regex is not None:
                gen_f = lambda: guidance.regex(pattern=regex)  # type: ignore
            else:
                gen_f = lambda: guidance.gen(
                    max_tokens=kwargs.get("max_tokens", 20), stop=["\n"]
                )  # type: ignore

            @guidance(stateless=True, dedent=False)  # type: ignore
            def make_predictions(lm, values, gen_f) -> guidance.models.Model:
                # gen_str = "\n".join([f"{value} -> {guidance.capture(guidance.gen(max_tokens=5), name=value)}" for value in values])
                gen_str = "\n".join(
                    [
                        f"{value} -> {guidance.capture(gen_f(), name=value)}"
                        for value in values
                    ]
                )
                return lm + gen_str

            example_str = "\n\nExamples:"
            for example in few_shot_examples:
                example_str += example.to_string(include_values=False)
                for k, v in example.mapping.items():
                    example_str += f"\n{k} -> {v}"
                example_str += "\n\n---"

            for i in range(0, len(sorted_values), batch_size):
                curr_batch_values = sorted_values[i : i + batch_size]
                current_batch_example = copy.deepcopy(current_example)
                current_batch_example.values = [str(i) for i in curr_batch_values]
                current_example_str = current_example.to_string(
                    include_values=False, list_options=list_options_in_prompt
                )

                # First check - do we need to load the model?
                in_cache = False
                if model.caching:
                    responses, key = model.check_cache(
                        CONSTRAINED_MAIN_INSTRUCTION,
                        example_str,
                        current_example_str,
                        current_batch_example.values,
                        funcs=[make_predictions, gen_f],
                    )
                    if responses is not None:
                        lm._variables.update(responses)  # type: ignore
                        in_cache = True
                if not in_cache:
                    lm: guidance.models.Model = maybe_load_lm(model, lm)
                with guidance.user():
                    lm += CONSTRAINED_MAIN_INSTRUCTION
                    lm += example_str
                    batch_lm = lm + current_example_str

                # TODO: since guidance does prefix caching, we don't actually reuse prompt tokens across batches
                model.prompt_tokens += len(
                    model.tokenizer.encode(batch_lm._current_prompt())
                )

                if not in_cache:
                    model.num_generation_calls += 1
                    with guidance.assistant():
                        batch_lm += make_predictions(
                            values=current_batch_example.values, gen_f=gen_f
                        )  # type: ignore
                        generated_batch_variables = {
                            k: batch_lm.get(k) for k in current_batch_example.values
                        }
                        lm._variables.update(generated_batch_variables)
                    if model.caching:
                        model.cache[key] = generated_batch_variables  # type: ignore
            lm_mapping: List[str] = [lm[value] for value in values]  # type: ignore
            model.completion_tokens += sum(
                [len(model.tokenizer.encode(v)) for v in lm_mapping]
            )
            mapped_values = cast_responses_to_datatypes(lm_mapping)
        else:
            messages_list: List[List[dict]] = []
            batch_sizes: List[int] = []
            for i in range(0, len(sorted_values), batch_size):
                messages = []
                curr_batch_values = sorted_values[i : i + batch_size]
                batch_sizes.append(len(curr_batch_values))
                current_batch_example = copy.deepcopy(current_example)
                current_batch_example.values = curr_batch_values
                messages.append(user(UNCONSTRAINED_MAIN_INSTRUCTION))
                # Add few-shot examples
                for example in few_shot_examples:
                    messages.append(user(example.to_string()))
                    messages.append(
                        assistant(CONST.DEFAULT_ANS_SEP.join(example.mapping.values()))
                    )
                # Add the current question + context for inference
                messages.append(
                    user(
                        current_batch_example.to_string(
                            list_options=list_options_in_prompt
                        )
                    )
                )
                messages_list.append(messages)

            responses: List[str] = model.generate(
                messages_list=messages_list, max_tokens=kwargs.get("max_tokens", None)
            )

            # Post-process language model response
            mapped_values = []
            total_missing_values = 0
            for idx, r in enumerate(responses):
                expected_len = batch_sizes[idx]
                predictions: List[Union[str, None]] = r.split(CONST.DEFAULT_ANS_SEP)  # type: ignore
                while len(predictions) < expected_len:
                    total_missing_values += 1
                    predictions.append(None)
                # Try to map to booleans, `None`, and numeric datatypes
                mapped_values.extend(cast_responses_to_datatypes(predictions))

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
