import copy
from typing import Dict, Union, Optional, Set, List
from guidance import gen, select
import pandas as pd
import re
from blendsql.models._model import Model
from blendsql._program import Program
from blendsql.ingredients.ingredient import QAIngredient
from blendsql.db.utils import single_quote_escape


class QAProgram(Program):
    def __call__(
        self,
        question: str,
        serialized_db: str,
        options: List[str] = None,
        long_answer: bool = False,
        table_title: str = None,
        **kwargs,
    ):
        _model = self.model
        with self.systemcontext:
            _model += "Answer the question for the table. "
            if long_answer:
                _model += "Make the answer as concrete as possible, providing more context and reasoning using the entire table.\n"
            else:
                _model += "Keep the answers as short as possible, without leading context. For example, do not say 'The answer is 2', simply say '2'.\n"
            if options is not None:
                _model += "Your answer should be a selection from one of the following options:\n"
                int_prefix_options = []
                for _idx, option in enumerate(options):
                    int_prefix_option = f"{option}"
                    int_prefix_options.append(int_prefix_option)
                    _model += f"{int_prefix_option}\n"
                # Add in title case, since this helps with selection
                modified_option_to_original = {}
                _options = copy.deepcopy(options)
                for option in options:
                    option = str(option)
                    for modified_option in [option.title(), option.upper()]:
                        _options.add(modified_option)
                        modified_option_to_original[modified_option] = option
                options = _options
        with self.usercontext:
            _model += f"\n\nQuestion: {question}"
            if table_title is not None:
                _model += f"\n\nContext: \n Table Description: {table_title} \n {serialized_db}"
            else:
                _model += f"\n\nContext: \n {serialized_db}"
        with self.assistantcontext:
            if options is not None:
                _model += select(options=[str(i) for i in options], name="result")
            else:
                _model += gen(name="result", **self.gen_kwargs)
        if options:
            _model._variables["result"] = modified_option_to_original.get(
                _model._variables["result"], _model._variables["result"]
            )
        return _model


class LLMQA(QAIngredient):
    def run(
        self,
        question: str,
        model: Model,
        options: Optional[Set[str]] = None,
        context: Optional[pd.DataFrame] = None,
        value_limit: Optional[int] = None,
        table_to_title: Optional[Dict[str, str]] = None,
        long_answer: bool = False,
        **kwargs,
    ) -> Union[str, int, float]:
        if context is not None:
            if value_limit is not None:
                context = context.iloc[:value_limit]
        res = model.predict(
            program=QAProgram,
            options=options,
            question=question,
            serialized_db=context.to_string() if context is not None else "",
            long_answer=long_answer,
            table_title=None,
            **kwargs,
        )
        return "'{}'".format(
            single_quote_escape(re.sub(r"^\d+\):", "", res["result"]).strip().lower())
        )
