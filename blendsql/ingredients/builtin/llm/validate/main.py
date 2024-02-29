from typing import Dict, Union, Optional
import pandas as pd
from guidance import select

from blendsql.models._model import Model
from blendsql._program import Program
from blendsql.ingredients.ingredient import QAIngredient


class ValidateProgram(Program):
    def __call__(self, question: str, serialized_db: str, table_title: str = None):
        with self.systemcontext:
            self.model += "You are a database expert in charge of validating a claim given a context. Given a claim and associated database context, you will respond 'true' if the claim is factual given the context, and 'false' if not."
        with self.usercontext:
            self.model += f"Claim: {question}"
            if table_title:
                self.model += f"\nTable Description: {table_title}"
            self.model += f"\n{serialized_db}\n\nAnswer:"
        with self.assistantcontext:
            self.model += select(options=["true", "false"], name="result")
        return self.model


class LLMValidate(QAIngredient):
    def run(
        self,
        question: str,
        model: Model,
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
            program=ValidateProgram,
            claim=question,
            serialized_db=context.to_string() if context is not None else "",
            long_answer=long_answer,
            table_title=None,
            **kwargs,
        )
        return int(res["result"] == "true")
