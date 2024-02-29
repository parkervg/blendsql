import logging
from typing import Union, Iterable, Any, Dict, Optional, List

import pandas as pd
from colorama import Fore
from tqdm import tqdm
from textwrap import dedent
from guidance import gen

from blendsql.models._model import Model
from blendsql.models import OpenaiLLM
from ast import literal_eval
from blendsql import _constants as CONST
from blendsql.ingredients.ingredient import MapIngredient
from blendsql._program import Program


class MapProgram(Program):
    def __call__(
        self,
        question: str,
        values: List[str],
        sep: str,
        include_tf_disclaimer: bool = False,
        output_type: str = None,
        example_outputs: str = None,
        table_title: str = None,
        colname: str = None,
        **kwargs,
    ):
        with self.systemcontext:
            self.model += """Given a set of values from a database, answer the question row-by-row, in order."""
            if include_tf_disclaimer:
                self.model += " If the question can be answered with 'true' or 'false', select `t` for 'true' or `f` for 'false'."
            self.model += dedent(
                f"""
            The answer should be a list separated by '{sep}', and have {len(values)} items in total.
            When you have given all {len(values)} answers, stop responding.
            If a given value has no appropriate answer, give '-' as a response.
            """
            )
        with self.usercontext:
            if self.few_shot:
                self.model += dedent(
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
                self.model += dedent(
                    f"The following values come from the column '{colname}', in a table titled '{table_title}'."
                )
            self.model += dedent(f"""Q: {question}\nValues:\n""")
            for value in values:
                self.model += f"`{value}`\n"
            if output_type:
                self.model += f"\nOutput type: {output_type}"
            if example_outputs:
                self.model += f"\nHere are some example outputs: {example_outputs}\n"
            self.model += "\nA:"
        with self.assistantcontext:
            self.model += gen(name="result", **self.gen_kwargs)
        return self.model


class LLMMap(MapIngredient):
    def run(
        self,
        question: str,
        model: Model,
        values: List[str],
        value_limit: Union[int, None] = None,
        example_outputs: Optional[str] = None,
        output_type: Optional[str] = None,
        pattern: Optional[str] = None,
        table_to_title: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Iterable[Any]:
        """For each value in a given column, calls a Model and retrieves the output.

        Args:
            question: The question to map onto the values. Will also be the new column name
            model: The Model (blender) we will make calls to.
            value_limit: Optional limit on the number of values to pass to the Model
            example_outputs: If binary == False, this gives the Model an example of the output we expect.
            output_type: One of 'numeric', 'string', 'bool'
            pattern: Optional regex to constrain answer generation.
            table_to_title: Mapping from tablename to a title providing some more context.

        Returns:
            Iterable[Any] containing the output of the Model for each value.
        """
        # Unpack default kwargs
        tablename, colname = self.unpack_default_kwargs(**kwargs)
        # OpenAI endpoints can't use patterns
        pattern = None if isinstance(model, OpenaiLLM) else pattern
        if value_limit is not None:
            values = values[:value_limit]
        values = [value if not pd.isna(value) else "-" for value in values]
        table_title = None
        if table_to_title is not None:
            if tablename not in table_to_title:
                logging.debug(f"Tablename {tablename} not in given table_to_title!")
            else:
                table_title = table_to_title[tablename]
        split_results = []
        # Only use tqdm if we're in debug mode
        context_manager = (
            tqdm(
                range(0, len(values), CONST.VALUE_BATCH_SIZE),
                total=len(values) // CONST.VALUE_BATCH_SIZE,
                desc=f"Making calls to Model with batch_size {CONST.VALUE_BATCH_SIZE}",
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET),
            )
            if logging.DEBUG >= logging.root.level
            else range(0, len(values), CONST.VALUE_BATCH_SIZE)
        )

        for i in context_manager:
            answer_length = len(values[i : i + CONST.VALUE_BATCH_SIZE])
            max_tokens = answer_length * 15
            include_tf_disclaimer = False

            if pattern is not None:
                if pattern.startswith("(t|f"):
                    include_tf_disclaimer = True
                    # max_tokens = answer_length * 2
            elif isinstance(model, OpenaiLLM):
                include_tf_disclaimer = True

            res = model.predict(
                program=MapProgram,
                question=question,
                sep=CONST.DEFAULT_ANS_SEP,
                values=values[i : i + CONST.VALUE_BATCH_SIZE],
                example_outputs=example_outputs,
                output_type=output_type,
                include_tf_disclaimer=include_tf_disclaimer,
                table_title=table_title,
                gen_kwargs={"max_tokens": max_tokens, "regex": pattern},
                **kwargs,
            )
            _r = [
                i.strip()
                for i in res["result"]
                .strip(CONST.DEFAULT_ANS_SEP)
                .split(CONST.DEFAULT_ANS_SEP)
            ]
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
                for i in _r
            ]
            expected_len = len(values[i : i + CONST.VALUE_BATCH_SIZE])
            if len(_r) != expected_len:
                logging.debug(
                    Fore.YELLOW
                    + f"Mismatch between length of values and answers!\nvalues:{expected_len}, answers:{len(_r)}"
                    + Fore.RESET
                )
                logging.debug(_r)
            # Cut off, in case we over-predicted
            _r = _r[:expected_len]
            # Add, in case we under-predicted
            while len(_r) < expected_len:
                _r.append(None)
            split_results.extend(_r)
        for idx, i in enumerate(split_results):
            try:
                split_results[idx] = literal_eval(i)
            except (ValueError, SyntaxError):
                continue
        logging.debug(
            Fore.YELLOW + f"Finished with values {split_results[:10]}" + Fore.RESET
        )
        return split_results
