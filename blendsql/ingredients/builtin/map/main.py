import logging
from typing import Union, Iterable, Any, Dict, Optional, List, Callable, Tuple

import re
import json
import pandas as pd
from colorama import Fore
from tqdm import tqdm
import guidance

from blendsql._logger import logger
from blendsql.models import Model, LocalModel, RemoteModel
from ast import literal_eval
from blendsql import _constants as CONST
from blendsql.ingredients.ingredient import MapIngredient
from blendsql._program import Program
from blendsql._exceptions import IngredientException
from blendsql.ingredients.generate import generate, user, assistant
from blendsql.ingredients.few_shot import AnnotatedMapExample, MapExample

DEFAULT_MAP_FEW_SHOT: List[AnnotatedMapExample] = [
    AnnotatedMapExample(
        **{
            "question": "Total penalty count?",
            "column_name": "Penalties (P+P+S+S)",
            "table_name": "Biathlon World Championships 2013",
            "output_type": "integer",
            "example_outputs": ["12", "3"],
            "examples": {
                "1 (0+0+0+1)": "1",
                "10 (5+3+2+0)": "10",
                "6 (2+2+2+0)": "6",
            },
        }
    ),
    AnnotatedMapExample(
        **{
            "question": "Is the time less than a week?",
            "column_name": "Length of use",
            "table_name": "Crest Whitestrips",
            "output_type": "boolean",
            "example_outputs": ["t", "f"],
            "examples": {"14 days": "f", "10 days": "f", "daily": "t", "2 hours": "t"},
        }
    ),
]
MAIN_INSTRUCTION = f"Given a set of values from a database, answer the question row-by-row, in order.\nYour outputs should be seperated by ';'."
OPTIONS_INSTRUCTION = "Your responses MUST select from one of the following values:\n"


class MapProgram(Program):
    def __call__(
        self,
        model: Model,
        current_example: MapExample,
        few_shot_examples: List[AnnotatedMapExample] = None,
        list_options_in_prompt: bool = True,
        max_tokens: Optional[int] = None,
        regex: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, str]:
        if few_shot_examples is None:
            few_shot_examples = DEFAULT_MAP_FEW_SHOT
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
                    for k, v in example.examples.items():
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
                    assistant(CONST.DEFAULT_ANS_SEP.join(example.examples.values()))
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


class LLMMap(MapIngredient):
    DESCRIPTION = """
    If question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column using the scalar function:
        `{{LLMMap('question', 'table::column')}}`
    """

    def run(
        self,
        model: Model,
        question: str,
        values: List[str],
        few_shot_examples: List[AnnotatedMapExample] = None,
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

            mapped_values: List[str] = model.predict(
                program=MapProgram,
                current_example=MapExample(
                    **{
                        "question": question,
                        "column_name": column_name,
                        "table_name": table_name,
                        "output_type": output_type,
                        "example_outputs": example_outputs,
                        "values": curr_batch_values,
                    }
                ),
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
