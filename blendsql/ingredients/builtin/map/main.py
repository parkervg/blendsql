import logging
from typing import Union, Iterable, Any, Dict, Optional, List, Callable, Tuple
from pathlib import Path
import re
import json
import pandas as pd
from colorama import Fore
from tqdm import tqdm
from attr import attrs, attrib
import guidance

from blendsql._logger import logger
from blendsql.models import Model, LocalModel, RemoteModel
from ast import literal_eval
from blendsql import _constants as CONST
from blendsql.ingredients.ingredient import MapIngredient
from blendsql._program import Program
from blendsql._exceptions import IngredientException
from blendsql.ingredients.generate import generate, user, assistant
from blendsql.ingredients.utils import initialize_retriever, partialclass
from .examples import AnnotatedMapExample, MapExample

DEFAULT_MAP_FEW_SHOT: List[AnnotatedMapExample] = [
    AnnotatedMapExample(**d)
    for d in json.loads(
        open(Path(__file__).resolve().parent / "./default_examples.json", "r").read()
    )
]
MAIN_INSTRUCTION = f"Given a set of values from a database, answer the question row-by-row, in order.\nYour outputs should be seperated by ';'."
OPTIONS_INSTRUCTION = "Your responses MUST select from one of the following values:\n"


class MapProgram(Program):
    def __call__(
        self,
        model: Model,
        current_example: MapExample,
        few_shot_examples: List[AnnotatedMapExample],
        list_options_in_prompt: bool = True,
        max_tokens: Optional[int] = None,
        regex: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, str]:
        if isinstance(model, LocalModel):
            options = current_example.options
            if all(x is not None for x in [options, regex]):
                raise IngredientException(
                    "MapIngredient exception!\nCan't have both `options` and `regex` argument passed."
                )
            if options:
                regex = f"({'|'.join([re.escape(option) for option in options])})"
            lm: guidance.models.Model = model.model_obj
            with guidance.user():
                lm += MAIN_INSTRUCTION
                lm += "\n\nExamples:"
                for example in few_shot_examples:
                    lm += example.to_string(include_values=False)
                    for k, v in example.mapping.items():
                        lm += f"\n{k} -> {v}"
                    lm += "\n\n---"
                lm += current_example.to_string(include_values=False)
            prompt = lm._current_prompt()
            if isinstance(model, LocalModel) and regex is not None:
                gen_f = lambda: guidance.regex(pattern=regex)
            else:
                gen_f = lambda: guidance.gen(max_tokens=max_tokens or 20)

            @guidance(stateless=True, dedent=False)
            def make_predictions(lm, values, gen_f) -> guidance.models.Model:
                for _idx, value in enumerate(values):
                    with guidance.user():
                        lm += f"\n{value} -> "
                    with guidance.assistant():
                        lm += guidance.capture(gen_f(), name=value)
                return lm

            lm += make_predictions(values=current_example.values, gen_f=gen_f)
            mapped_values = [lm[value] for value in current_example.values]
        else:
            # Use the 'old' style of prompting when we have a remote model
            messages = []
            messages.append(user(MAIN_INSTRUCTION))
            # Add few-shot examples
            for example in few_shot_examples:
                messages.append(user(example.to_string()))
                messages.append(
                    assistant(CONST.DEFAULT_ANS_SEP.join(example.mapping.values()))
                )
            # Add the current question + context for inference
            messages.append(user(current_example.to_string()))
            response = generate(model, messages=messages, max_tokens=max_tokens or 1000)
            # Post-process language model response
            mapped_values = [
                i.strip()
                for i in response.strip(CONST.DEFAULT_ANS_SEP).split(
                    CONST.DEFAULT_ANS_SEP
                )
            ]
            prompt = "".join([i["content"] for i in messages])
        return mapped_values, prompt


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
    batch_size: int = attrib(default=5)

    @classmethod
    def from_args(
        cls,
        model: Optional[Model] = None,
        few_shot_examples: Optional[List[dict]] = None,
        list_options_in_prompt: bool = True,
        batch_size: Optional[int] = 5,
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
            from blendsql import blend, LLMMap
            from blendsql.ingredients.builtin import DEFAULT_MAP_FEW_SHOT

            ingredients = {
                LLMMap.from_args(
                    few_shot_examples=[
                        *DEFAULT_MAP_FEW_SHOT,
                        {
                            "question": "Is this a sport?",
                            "mapping": {
                                "Soccer": "t",
                                "Chair": "f",
                                "Banana": "f",
                                "Golf": "t"
                            },
                            # Below are optional
                            "column_name": "Items",
                            "table_name": "Table",
                            "example_outputs": ["t", "f"],
                            "options": ["t", "f"],
                            "output_type": "boolean"
                        }
                    ],
                    # Will fetch `k` most relevant few-shot examples using embedding-based retriever
                    k=2,
                    # How many inference values to pass to model at once
                    batch_size=5,
                )
            }
            smoothie = blend(
                query=blendsql,
                db=db,
                ingredients=ingredients,
                default_model=model,
            )
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
        few_shot_retriever: Callable[[str], List[AnnotatedMapExample]] = None,
        options: List[str] = None,
        list_options_in_prompt: bool = None,
        value_limit: Union[int, None] = None,
        example_outputs: Optional[str] = None,
        output_type: Optional[str] = None,
        regex: Optional[Callable[[int], str]] = None,
        table_to_title: Optional[Dict[str, str]] = None,
        batch_size: int = None,
        **kwargs,
    ) -> Iterable[Any]:
        """For each value in a given column, calls a Model and retrieves the output.

        Args:
            question: The question to map onto the values. Will also be the new column name
            model: The Model (blender) we will make calls to.
            values: The list of values to apply question to.
            value_limit: Optional limit on the number of values to pass to the Model
            example_outputs: If binary == False, this gives the Model an example of the output we expect.
            output_type: One of 'numeric', 'string', 'bool'
            regex: Optional regex to constrain answer generation.
            table_to_title: Mapping from tablename to a title providing some more context.

        Returns:
            Iterable[Any] containing the output of the Model for each value.
        """
        if model is None:
            raise IngredientException(
                "LLMMap requires a `Model` object, but nothing was passed!\nMost likely you forgot to set the `default_model` argument in `blend()`"
            )
        if batch_size is None:
            batch_size = CONST.MAP_BATCH_SIZE
        if few_shot_retriever is None:
            few_shot_retriever = lambda *_: DEFAULT_MAP_FEW_SHOT
        # Unpack default kwargs
        table_name, column_name = self.unpack_default_kwargs(**kwargs)
        # Remote endpoints can't use patterns
        regex = None if isinstance(model, RemoteModel) else regex
        if value_limit is not None:
            values = values[:value_limit]
        values = [value if not pd.isna(value) else "-" for value in values]
        split_results: List[Union[str, None]] = []
        # Only use tqdm if we're in debug mode
        context_manager: Iterable = (
            tqdm(
                range(0, len(values), batch_size),
                total=len(values) // batch_size,
                desc=f"Making calls to Model with batch_size {batch_size}",
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET),
            )
            if logger.level <= logging.DEBUG
            else range(0, len(values), batch_size)
        )

        for i in context_manager:
            answer_length = len(values[i : i + batch_size])
            max_tokens = answer_length * 15
            curr_batch_values = values[i : i + batch_size]
            current_example = MapExample(
                **{
                    "question": question,
                    "column_name": column_name,
                    "table_name": table_name,
                    "output_type": output_type,
                    "example_outputs": example_outputs,
                    "values": curr_batch_values,
                }
            )
            few_shot_examples: List[AnnotatedMapExample] = few_shot_retriever(
                current_example.to_string()
            )
            mapped_values: List[str] = model.predict(
                program=MapProgram,
                current_example=current_example,
                question=question,
                few_shot_examples=few_shot_examples,
                options=options,
                list_options_in_prompt=list_options_in_prompt,
                example_outputs=example_outputs,
                output_type=output_type,
                table_name=table_name,
                column_name=column_name,
                regex=regex,
                max_tokens=max_tokens,
                **kwargs,
            )
            # Try to map to booleans and `None`
            mapped_values = [
                {
                    "t": True,
                    "f": False,
                    "true": True,
                    "false": False,
                    "y": True,
                    "n": False,
                    "yes": True,
                    "no": False,
                    CONST.DEFAULT_NAN_ANS: None,
                }.get(i.lower(), i)
                for i in mapped_values
            ]
            expected_len = len(curr_batch_values)
            if len(mapped_values) != expected_len:
                logger.debug(
                    Fore.YELLOW
                    + f"Mismatch between length of values and answers!\nvalues:{expected_len}, answers:{len(mapped_values)}"
                    + Fore.RESET
                )
                logger.debug(mapped_values)
            split_results.extend(mapped_values)
        for idx, i in enumerate(split_results):
            if i is None:
                continue
            if isinstance(i, str):
                i = i.replace(",", "")
            try:
                split_results[idx] = literal_eval(i)
                assert isinstance(i, (float, int, str))
            except (ValueError, SyntaxError, AssertionError):
                continue
        logger.debug(
            Fore.YELLOW
            + f"Finished LLMMap with values:\n{json.dumps(dict(zip(values[:10], split_results[:10])), indent=4)}"
            + Fore.RESET
        )
        return split_results
