import copy
from typing import Dict, Union, Optional, Set, Tuple
import pandas as pd
import re

from blendsql.models import Model, OllamaLLM
from blendsql._exceptions import InvalidBlendSQL
from blendsql._program import Program
from blendsql.ingredients.ingredient import QAIngredient
from blendsql.db.utils import single_quote_escape
from blendsql import generate
from blendsql._exceptions import IngredientException


class QAProgram(Program):
    def __call__(
        self,
        model: Model,
        question: str,
        context: Optional[pd.DataFrame] = None,
        options: Optional[Set[str]] = None,
        long_answer: Optional[bool] = False,
        table_title: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Tuple[str, str]:
        prompt = ""
        serialized_db = context.to_string() if context is not None else ""
        prompt += "Answer the question for the table. "
        modified_option_to_original = {}
        if long_answer:
            prompt += "Make the answer as concrete as possible, providing more context and reasoning using the entire table.\n"
        else:
            prompt += "Keep the answers as short as possible, without leading context. For example, do not say 'The answer is 2', simply say '2'.\n"
        if options is not None:
            # Add in title case, since this helps with selection
            _options = copy.deepcopy(options)
            # Below we check to see if our options have a unique first word
            # sometimes, the model will generate 'Frank' instead of 'Frank Smith'
            # We still want to align that, in this case
            add_first_word = False
            if len(set([i.split(" ")[0] for i in options])) == len(options):
                add_first_word = True
            for option in options:
                option = str(option)
                for modified_option in [option.title(), option.upper()]:
                    _options.add(modified_option)
                    modified_option_to_original[modified_option] = option
                if add_first_word:
                    modified_option_to_original[option.split(" ")[0]] = option
            options = _options
        prompt += f"\n\nQuestion: {question}"
        if table_title is not None:
            prompt += (
                f"\n\nContext: \n Table Description: {table_title} \n {serialized_db}"
            )
        else:
            prompt += f"\n\nContext: \n {serialized_db}"
        if options is not None:
            if isinstance(model, OllamaLLM):
                raise InvalidBlendSQL(
                    "Can't use `options` argument in LLMQA with an Ollama model!"
                )
            _response = generate.choice(
                model, prompt=prompt, choices=[re.escape(str(i)) for i in options]
            )
            # Map from modified options to original, as they appear in DB
            response: str = modified_option_to_original.get(_response, _response)
        else:
            response = generate.text(
                model, prompt=prompt, max_tokens=max_tokens, stop_at="\n"
            )
        return (response, prompt)


class LLMQA(QAIngredient):
    DESCRIPTION = """
    If mapping to a new column still cannot answer the question with valid SQL, turn to an end-to-end solution using the aggregate function:
        `{{LLMQA('question', (blendsql))}}`
        Optionally, this function can take an `options` argument to restrict its output to an existing SQL column.
        For example: `... WHERE column = {{LLMQA('question', (blendsql), options='table::column)}}`
    """

    def run(
        self,
        model: Model,
        question: str,
        options: Optional[Set[str]] = None,
        context: Optional[pd.DataFrame] = None,
        value_limit: Optional[int] = None,
        table_to_title: Optional[Dict[str, str]] = None,
        long_answer: bool = False,
        **kwargs,
    ) -> Union[str, int, float]:
        if model is None:
            raise IngredientException(
                "LLMQA requires a `Model` object, but nothing was passed!\nMost likely you forgot to set the `default_model` argument in `blend()`"
            )
        if context is not None:
            if value_limit is not None:
                context = context.iloc[:value_limit]
        result = model.predict(
            program=QAProgram,
            options=options,
            question=question,
            context=context,
            long_answer=long_answer,
            table_title=None,
            **kwargs,
        )
        # Post-process language model response
        return "'{}'".format(single_quote_escape(result.strip()))
