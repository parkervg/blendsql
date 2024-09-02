import copy
from typing import Dict, Union, Optional, Set, Tuple
import pandas as pd
import guidance
from colorama import Fore

from blendsql.models import Model, LocalModel
from blendsql.ingredients.generate import generate
from blendsql._program import Program
from blendsql.ingredients.ingredient import QAIngredient
from blendsql.db.utils import single_quote_escape
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
        if isinstance(model, LocalModel):
            m: guidance.models.Model = model.model_obj
        else:
            m: str = ""
        serialized_db = context.to_string() if context is not None else ""
        m += "Answer the question for the table. "
        options_alias_to_original = {}
        if long_answer:
            m += "Make the answer as concrete as possible, providing more context and reasoning using the entire table.\n"
        else:
            m += "Keep the answers as short as possible, without leading context. For example, do not say 'The answer is 2', simply say '2'.\n"
        if options is not None:
            # Add in title case, since this helps with selection
            options_with_aliases = copy.deepcopy(options)
            # Below we check to see if our options have a unique first word
            # sometimes, the model will generate 'Frank' instead of 'Frank Smith'
            # We still want to align that, in this case
            add_first_word = False
            if len(set([i.split(" ")[0] for i in options])) == len(options):
                add_first_word = True
            for option in options:
                option = str(option)
                for option_alias in [option.title(), option.upper()]:
                    options_with_aliases.add(option_alias)
                    options_alias_to_original[option_alias] = option
                if add_first_word:
                    option_alias = option.split(" ")[0]
                    options_alias_to_original[option_alias] = option
                    options_with_aliases.add(option_alias)
        m += f"\n\nQuestion: {question}"
        if table_title is not None:
            m += f"\n\nContext: \n Table Description: {table_title} \n {serialized_db}"
        else:
            m += f"\n\nContext: \n {serialized_db}"
        if options and not isinstance(model, LocalModel):
            m += f"\n\nFor your answer, select from one of the following options: {options}"
        m += "\n\nAnswer:\n"
        if isinstance(model, LocalModel):
            prompt = m._current_prompt()
            if options is not None:
                response = (
                    m
                    + guidance.capture(
                        guidance.select(options=options_with_aliases),
                        name="response",
                    )
                )._variables["response"]
            else:
                response = (
                    m
                    + guidance.capture(
                        guidance.gen(max_tokens=max_tokens or 50), name="response"
                    )
                )._variables["response"]
        else:
            prompt = m
            if model.tokenizer is not None:
                max_tokens = (
                    max(
                        [
                            len(model.tokenizer.encode(alias))
                            for alias in options_alias_to_original
                        ]
                    )
                    if options
                    else max_tokens
                )
            response = generate(
                model,
                prompt=prompt,
                options=options,
                max_tokens=max_tokens,
                stop_at=["\n"],
            )
        # Map from modified options to original, as they appear in DB
        response: str = options_alias_to_original.get(response, response)
        if options and response not in options:
            print(
                Fore.RED
                + f"Model did not select from a valid option!\nExpected one of {options}, got {response}"
                + Fore.RESET
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
