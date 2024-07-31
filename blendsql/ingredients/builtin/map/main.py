from typing import Union, Iterable, Any, Dict, Optional, List, Callable, Tuple
import pandas as pd
from functools import partial

from blendsql.utils import newline_dedent
from blendsql.ingredients.utils import batch_run_map
from blendsql._logger import logger
from blendsql.models import Model, LocalModel, RemoteModel, OpenaiLLM
from blendsql import _constants as CONST
from blendsql.ingredients.ingredient import MapIngredient
from blendsql._program import Program
from blendsql import generate
from blendsql._exceptions import IngredientException


class MapProgram(Program):
    def __call__(
        self,
        model: Model,
        question: str,
        values: List[str],
        sep: str,
        include_tf_disclaimer: bool = False,
        max_tokens: Optional[int] = None,
        regex: Optional[Callable[[int], str]] = None,
        output_type: Optional[str] = None,
        example_outputs: Optional[str] = None,
        table_title: Optional[str] = None,
        colname: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, str]:
        prompt = ""
        prompt += """Given a set of values from a database, answer the question row-by-row, in order."""
        if include_tf_disclaimer:
            prompt += " If the question can be answered with 'true' or 'false', select `t` for 'true' or `f` for 'false'."
        prompt += newline_dedent(
            f"""
        The answer should be a list separated by '{sep}', and have {len(values)} items in total.
        When you have given all {len(values)} answers, stop responding.
        If a given value has no appropriate answer, give '-' as a response.
        """
        )
        prompt += newline_dedent(
            """
        ---

        The following values come from the column 'Home Town', in a table titled '2010\u201311 North Carolina Tar Heels men's basketball team'.
        Q: What state is this?
        Values:
        `Ames, IA`
        `Carrboro, NC`
        `Kinston, NC`
        `Encino, CA`

        Output type: string
        Here are some example outputs: `MA;CA;-;`

        A: IA;NC;NC;CA

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

        The following values come from the column 'term', in a table titled 'Electoral district of Lachlan'.
        Q: how long did it last?
        Values:
        `1859–1864`
        `1864–1869`
        `1869–1880`

        Output type: numeric

        A: 5;5;11

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
        prompt += newline_dedent(f"""Q: {question}\nValues:\n""")
        for value in values:
            prompt += f"`{value}`\n"
        if output_type:
            prompt += f"\nOutput type: {output_type}"
        if example_outputs:
            prompt += f"\nHere are some example outputs: {example_outputs}\n"
        prompt += "\nA:"
        if isinstance(model, LocalModel) and regex is not None:
            response = generate.regex(model, prompt=prompt, regex=regex(len(values)))
        else:
            response = generate.text(
                model, prompt=prompt, max_tokens=max_tokens, stop_at="\n"
            )
        return (response, prompt)


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
        include_tf_disclaimer = False
        if output_type == "boolean":
            include_tf_disclaimer = True
        elif isinstance(model, OpenaiLLM):
            include_tf_disclaimer = True
        pred_func = partial(
            model.predict,
            program=MapProgram,
            question=question,
            sep=CONST.DEFAULT_ANS_SEP,
            example_outputs=example_outputs,
            output_type=output_type,
            include_tf_disclaimer=include_tf_disclaimer,
            table_title=table_title,
            regex=regex,
            **kwargs,
        )
        split_results: List[Any] = batch_run_map(
            pred_func,
            values=values,
            batch_size=CONST.MAP_BATCH_SIZE,
            sep=CONST.DEFAULT_ANS_SEP,
            nan_answer=CONST.DEFAULT_NAN_ANS,
        )
        return split_results
