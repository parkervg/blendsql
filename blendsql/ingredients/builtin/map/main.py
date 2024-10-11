import logging
from typing import Union, Iterable, Any, Dict, Optional, List, Callable, Tuple

import re
import json
import pandas as pd
from colorama import Fore
from tqdm import tqdm
import guidance

from blendsql.utils import newline_dedent
from blendsql._logger import logger
from blendsql.models import Model, LocalModel, RemoteModel, OpenaiLLM
from ast import literal_eval
from blendsql import _constants as CONST
from blendsql.ingredients.ingredient import MapIngredient
from blendsql._program import Program
from blendsql._exceptions import IngredientException
from blendsql.ingredients.generate import generate


class MapProgram(Program):
    def __call__(
        self,
        model: Model,
        question: str,
        values: List[str],
        sep: str,
        options: List[str] = None,
        allow_null_option: bool = True,
        list_options_in_prompt: bool = True,
        include_tf_disclaimer: bool = False,
        max_tokens: Optional[int] = None,
        regex: Optional[str] = None,
        output_type: Optional[str] = None,
        example_outputs: Optional[str] = None,
        table_title: Optional[str] = None,
        colname: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, str]:
        if isinstance(model, LocalModel):
            if all(x is not None for x in [options, regex]):
                raise IngredientException(
                    "MapIngredient exception!\nCan't have both `options` and `regex` argument passed."
                )
            if options:
                regex = f"({'|'.join([re.escape(option) for option in options])})"
            m: guidance.models.Model = model.model_obj
            with guidance.system():
                m += """Given a set of values from a database, answer the question row-by-row, in order."""
                if include_tf_disclaimer:
                    m += " If the question can be answered with 'true' or 'false', select `t` for 'true' or `f` for 'false'."
                m += newline_dedent(
                    """
                ---
    
                The following values come from the column 'Penalties (P+P+S+S)', in a table titled 'Biathlon World Championships 2013 \u2013 Men's pursuit'.
                Q: Total penalty count?
                Here are some example outputs: '1', '2', '5'
                A:
                    - 1 (0+0+0+1) -> 1
                    - 10 (5+3+2+0) -> 10
                    - 6 (2+2+2+0) -> 6
    
                ---
    
                The following values come from the column 'Length of use', in a table titled 'Crest Whitestrips'.
                Q: Is the time less than a week?
                A:
                    - 14 days -> f
                    - 10 days -> f
                    - daily -> t
                    - 2 hours -> t
    
                ---
                """
                )
                if table_title:
                    m += newline_dedent(
                        f"The following values come from the column '{colname}', in a table titled '{table_title}'."
                    )
            with guidance.user():
                if list_options_in_prompt and options:
                    m += f"Your responses should select from one of the following values:\n"
                    m += "\n".join(options)
                    m += "\n\n"
                m += newline_dedent(f"""Q: {question}\nA:\n""")
            prompt = m._current_prompt()
            if isinstance(model, LocalModel) and regex is not None:
                gen_f = lambda: guidance.regex(pattern=regex)
            else:
                gen_f = lambda: guidance.gen(max_tokens=max_tokens or 1000)

            @guidance(stateless=True, dedent=False)
            def make_predictions(lm, values, gen_f) -> guidance.models.Model:
                for _idx, value in enumerate(values):
                    lm += f"\n{value} -> " + guidance.capture(gen_f(), name=value)
                return lm

            with guidance.assistant():
                m += make_predictions(values=values, gen_f=gen_f)
            print(prompt)
            return ([m[value] for value in values], prompt)
        else:
            # Use the 'old' style of prompting when we have a remote model
            prompt = ""
            prompt += """Given a set of values from a database, answer the question row-by-row, in order."""
            if include_tf_disclaimer:
                prompt += " If the question can be answered with 'true' or 'false', select `t` for 'true' or `f` for 'false'."
            prompt += newline_dedent(
                f"""
                    The answer should be a list separated by '{sep}', and have {len(values)} items in total.
                    When you have given all {len(values)} answers, stop responding.
                    """
            )
            prompt += newline_dedent(
                """
            ---

            The following values come from the column 'Penalties (P+P+S+S)', in a table titled 'Biathlon World Championships 2013 \u2013 Men's pursuit'.
            Q: Total penalty count?
            Values:
            `1 (0+0+0+1)`
            `10 (5+3+2+0)`
            `6 (2+2+2+0)`

            Output type: numeric
            Here are some example outputs: `9;-`

            A: 1;10;6

            ---

            The following values come from the column 'Length of use', in a table titled 'Crest Whitestrips'.
            Q: Is the time less than a week?
            Values:
            `14 days`
            `10 days`
            `daily`
            `2 hours`

            Output type: boolean
            A: f;f;t;t

            ---
            """
            )
            if table_title:
                prompt += newline_dedent(
                    f"The following values come from the column '{colname}', in a table titled '{table_title}'."
                )
            if list_options_in_prompt and options:
                prompt += (
                    f"Your responses should select from one of the following values:\n"
                )
                prompt += "\n".join(options)
                prompt += "\n\n"
            prompt += newline_dedent(f"""Q: {question}\nValues:\n""")
            for value in values:
                prompt += f"`{value}`\n"
            if output_type:
                prompt += f"\nOutput type: {output_type}"
            if example_outputs:
                prompt += f"\nHere are some example outputs: {example_outputs}\n"
            prompt += "\nA:"
            response = generate(model, prompt=prompt, max_tokens=max_tokens or 1000)
            # Post-process language model response
            _r = [
                i.strip()
                for i in response.strip(CONST.DEFAULT_ANS_SEP).split(
                    CONST.DEFAULT_ANS_SEP
                )
            ]
            return (_r, prompt)


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
        options: List[str] = None,
        allow_null_option: bool = None,
        list_options_in_prompt: bool = None,
        value_limit: Union[int, None] = None,
        example_outputs: Optional[str] = None,
        output_type: Optional[str] = None,
        regex: Optional[Callable[[int], str]] = None,
        table_to_title: Optional[Dict[str, str]] = None,
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
        # Unpack default kwargs
        tablename, colname = self.unpack_default_kwargs(**kwargs)
        # Remote endpoints can't use patterns
        regex = None if isinstance(model, RemoteModel) else regex
        if value_limit is not None:
            values = values[:value_limit]
        values = [value if not pd.isna(value) else "-" for value in values]
        table_title = None
        if table_to_title is not None:
            if tablename not in table_to_title:
                logger.debug(f"Tablename {tablename} not in given table_to_title!")
            else:
                table_title = table_to_title[tablename]
        split_results: List[Union[str, None]] = []
        # Only use tqdm if we're in debug mode
        context_manager: Iterable = (
            tqdm(
                range(0, len(values), CONST.MAP_BATCH_SIZE),
                total=len(values) // CONST.MAP_BATCH_SIZE,
                desc=f"Making calls to Model with batch_size {CONST.MAP_BATCH_SIZE}",
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET),
            )
            if logger.level <= logging.DEBUG
            else range(0, len(values), CONST.MAP_BATCH_SIZE)
        )

        for i in context_manager:
            answer_length = len(values[i : i + CONST.MAP_BATCH_SIZE])
            max_tokens = answer_length * 15
            include_tf_disclaimer = False
            curr_batch_values = values[i : i + CONST.MAP_BATCH_SIZE]

            if output_type == "boolean":
                include_tf_disclaimer = True
            elif isinstance(model, OpenaiLLM):
                include_tf_disclaimer = True

            result: List[str] = model.predict(
                program=MapProgram,
                question=question,
                sep=CONST.DEFAULT_ANS_SEP,
                values=curr_batch_values,
                options=options,
                allow_null_option=allow_null_option,
                list_options_in_prompt=list_options_in_prompt,
                example_outputs=example_outputs,
                output_type=output_type,
                include_tf_disclaimer=include_tf_disclaimer,
                table_title=table_title,
                regex=regex,
                max_tokens=max_tokens,
                **kwargs,
            )
            # Try to map to booleans and `None`
            _r = [
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
                for i in result
            ]
            expected_len = len(curr_batch_values)
            if len(_r) != expected_len:
                logger.debug(
                    Fore.YELLOW
                    + f"Mismatch between length of values and answers!\nvalues:{expected_len}, answers:{len(_r)}"
                    + Fore.RESET
                )
                logger.debug(_r)
            split_results.extend(_r)
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
