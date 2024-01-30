import logging
from typing import Union, Iterable, Any, Dict

import pandas as pd
from colorama import Fore
from tqdm import tqdm

from blendsql.ingredients.builtin.llm.utils import (
    OpenaiEndpoint,
    construct_gen_clause,
)
from blendsql.ingredients.builtin.llm.endpoint import Endpoint
from ast import literal_eval
from blendsql import _constants as CONST
from blendsql.ingredients.ingredient import MapIngredient, IngredientException
from blendsql import _programs as programs


class LLMMap(MapIngredient):
    def run(
        self,
        question: str,
        endpoint: Endpoint,
        value_limit: Union[int, None] = None,
        example_outputs: str = None,
        output_type: str = None,
        pattern: str = None,
        table_to_title: Dict[str, str] = None,
        **kwargs,
    ) -> Iterable[Any]:
        """For each value in a given column, calls an OpenAI LLM endpoint_name and retrieves the output.

        Args:
            question: The question to map onto the values. Will also be the new column name
            value_limit: Optional limit on the number of values to pass to the LLM
            example_outputs: str, if binary == False, this gives the LLM an example of the output we expect.
            pattern: str, optional regex to constrain answer generation.
            endpoint_name: str, name of the OpenAI endpoint_name we will make calls to.

        Returns:
            Iterable[Any] containing the output of the LLM for each value.
        """
        if question is None:
            raise IngredientException("Need to specify `question` for LLMmap")
        # Unpack default kwargs
        values, original_table, tablename, colname = self.unpack_default_kwargs(
            **kwargs
        )
        # OpenAI endpoints can't use patterns
        pattern = None if isinstance(endpoint, OpenaiEndpoint) else pattern
        if value_limit is not None:
            values = values[:value_limit]
        values_dict = [
            {"value": value if not pd.isna(value) else "-", "idx": idx}
            for idx, value in enumerate(values)
        ]
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
                range(0, len(values_dict), CONST.VALUE_BATCH_SIZE),
                total=len(values_dict) // CONST.VALUE_BATCH_SIZE,
                desc=f"Making calls to LLM with batch_size {CONST.VALUE_BATCH_SIZE}",
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET),
            )
            if logging.DEBUG >= logging.root.level
            else range(0, len(values_dict), CONST.VALUE_BATCH_SIZE)
        )

        # examples = select_and_construct_map_examples(question=arg)

        for i in context_manager:
            answer_length = len(values_dict[i : i + CONST.VALUE_BATCH_SIZE])
            max_tokens = answer_length * 15
            include_tf_disclaimer = False

            if pattern is not None:
                if pattern.startswith("(t|f"):
                    include_tf_disclaimer = True
                    max_tokens = answer_length * 2
            elif isinstance(endpoint, OpenaiEndpoint):
                include_tf_disclaimer = True

            gen_clause: str = construct_gen_clause(
                pattern=pattern, max_tokens=max_tokens, **endpoint.gen_kwargs
            )
            program: str = (
                programs.MAP_PROGRAM_CHAT(gen_clause)
                if endpoint.endpoint_name in CONST.OPENAI_CHAT_LLM
                else programs.MAP_PROGRAM_COMPLETION(gen_clause)
            )
            res = endpoint.predict(
                program=program,
                question=question,
                sep=CONST.DEFAULT_ANS_SEP,
                answer_length=len(values_dict[i : i + CONST.VALUE_BATCH_SIZE]),
                values_dict=values_dict[i : i + CONST.VALUE_BATCH_SIZE],
                example_outputs=example_outputs,
                output_type=output_type,
                include_tf_disclaimer=include_tf_disclaimer,
                table_title=table_title,
                colname=colname,
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
            expected_len = len(values_dict[i : i + CONST.VALUE_BATCH_SIZE])
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
