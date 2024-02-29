from typing import Dict, Union, Optional, Set, List
from guidance import gen, select
import pandas as pd
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
        with self.systemcontext:
            self.model += "Answer the question for the table. "
            if long_answer:
                self.model += "Make the answer as concrete as possible, providing more context and reasoning using the entire table."
            else:
                self.model += "Keep the answers as short as possible, without leading context. For example, do not say 'The answer is 2', simply say '2'."
        with self.usercontext:
            self.model += f"Question: {question}"
            if table_title is not None:
                self.model = f"Table Description: {table_title}"
            self.model += f"\n\n {serialized_db}\n"
            if options is not None:
                self.model += "\nSelect from one of the following options.\n"
                for option in options:
                    self.model += f"- {option}\n"
        with self.assistantcontext:
            if options is not None:
                self.model += select(options=[str(i) for i in options], name="result")
            else:
                self.model += gen(name="result", **self.gen_kwargs)
        return self.model


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
        return "'{}'".format(single_quote_escape(res["result"].strip()))
