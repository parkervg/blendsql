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
    MapExample,
    AnnotatedMapExample,
    ConstrainedMapExample,
    ConstrainedAnnotatedMapExample,
    UnconstrainedMapExample,
    UnconstrainedAnnotatedMapExample,
)

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

    def __call__(
        self,
        question: t.Optional[str] = None,
        context: t.Optional[str] = None,
        options: t.Optional[t.Union[list, str]] = None,
        batch_size: t.Optional[int] = None,
        *args,
        **kwargs,
    ) -> tuple:
        return super().__call__(
            question=question,
            context=context,
            options=options,
            batch_size=batch_size,
            *args,
            **kwargs,
        )

    def run(
        self,
        model: Model,
        question: str,
        values: t.List[str],
        list_options_in_prompt: bool,
        few_shot_retriever: t.Optional[
            t.Callable[[str], t.List[AnnotatedMapExample]]
        ] = None,
        options: t.Optional[Collection[str]] = None,
        value_limit: t.Optional[int] = None,
        example_outputs: t.Optional[str] = None,
        output_type: t.Optional[t.Union[DataType, str]] = None,
        batch_size: int = None,
        **kwargs,
    ) -> t.List[t.Union[float, int, str, bool]]:
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
        resolved_output_type: DataType = prepare_datatype(
            output_type=output_type, options=options, modifier=None
        )
        current_example = MapExample(
            **{
                "question": question,
                "column_name": column_name,
                "table_name": table_name,
                "output_type": resolved_output_type,
                "example_outputs": example_outputs,
                "options": options,
                # Random subset of values for few-shot example retrieval
                # these will get replaced during batching later
                "values": values[:10],
            }
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
                for example in few_shot_retriever(current_example.to_string())
            ]
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
                gen_f = lambda _: guidance.select(options=options)  # type: ignore
            elif output_type == "substring":
                # Special case for substring datatypes
                gen_f = lambda s: guidance.substring(target_string=s)
            else:
                gen_f = lambda _: guidance.gen(
                    max_tokens=kwargs.get("max_tokens", 200),
                    stop=["\n\t"] + ['"']
                    if current_example.output_type.name == "str"
                    else [],
                    regex=regex,
                )  # type: ignore

            @guidance(stateless=True, dedent=False)  # type: ignore
            def make_predictions(
                lm, values, str_output: bool, gen_f
            ) -> guidance.models.Model:
                quotes = [
                    '"""' if any(c in value for c in ["\n", '"']) else '"'
                    for value in values
                ]
                gen_str = "\n".join(
                    [
                        f"""\t\tf({quote}{value}{quote}) == {'"' if str_output else ''}{guidance.capture(gen_f(value), name=value)}{'"' if str_output else ''}"""
                        for value, quote in zip(values, quotes)
                    ]
                )
                return lm + gen_str

            example_str = ""
            if len(few_shot_examples) > 0:
                for example in few_shot_examples:
                    example_str += example.to_string()

            loaded_lm = False
            # Due to guidance's prefix caching, this is a one-time cost
            model.prompt_tokens += len(
                model.tokenizer.encode(CONSTRAINED_MAIN_INSTRUCTION + example_str)
            )
            for i in range(0, len(sorted_values), batch_size):
                curr_batch_values = sorted_values[i : i + batch_size]
                current_batch_example = copy.deepcopy(current_example)
                current_batch_example.values = [str(i) for i in curr_batch_values]
                current_example_str = current_example.to_string(
                    list_options=list_options_in_prompt,
                    add_leading_newlines=False,
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
                if not in_cache and not loaded_lm:
                    lm: guidance.models.Model = maybe_load_lm(model, lm)
                    loaded_lm = True
                    with guidance.user():
                        lm += CONSTRAINED_MAIN_INSTRUCTION
                        lm += example_str

                with guidance.user():
                    batch_lm = lm + current_example_str

                model.prompt_tokens += len(model.tokenizer.encode(current_example_str))

                if not in_cache:
                    model.num_generation_calls += 1
                    with guidance.assistant():
                        batch_lm += make_predictions(
                            values=current_batch_example.values,
                            str_output=(current_example.output_type.name == "str"),
                            gen_f=gen_f,
                        )  # type: ignore
                        generated_batch_variables = {
                            k: batch_lm.get(k) for k in current_batch_example.values
                        }
                        lm._variables.update(generated_batch_variables)
                    if model.caching:
                        model.cache[key] = generated_batch_variables  # type: ignore
            lm_mapping: t.List[str] = [lm[value] for value in values]  # type: ignore
            model.completion_tokens += sum(
                [len(model.tokenizer.encode(v)) for v in lm_mapping]
            )
            # For each value, call the DataType's `coerce_fn()`
            mapped_values = [
                current_example.output_type.coerce_fn(s) for s in lm_mapping
            ]
        else:
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

            print("\n".join(responses))
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
                    [current_example.output_type.coerce_fn(s) for s in predictions]
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
