from typing import Dict, Union, Optional
import pandas as pd
from guidance import select

from blendsql.models._model import Model
from blendsql._program import Program
from blendsql.ingredients.ingredient import QAIngredient


class ValidateProgram(Program):
    def __call__(
        self,
        question: str,
        context: pd.DataFrame = None,
        table_title: str = None,
        **kwargs,
    ):
        serialized_db = context.to_string() if context is not None else ""
        _model = self.model
        with self.systemcontext:
            _model += "You are a database expert in charge of validating a claim given a context. Given a claim and associated database context, you will respond 'true' if the claim is factual given the context, and 'false' if not."
        with self.usercontext:
            _model += f"Claim: {question}"
            if table_title:
                _model += f"\nTable Description: {table_title}"
            _model += f"\n{serialized_db}\n\nAnswer:"
        with self.assistantcontext:
            _model += select(options=["true", "false"], name="result")
        return _model


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
            question=question,
            context=context,
            table_title=None,
            **kwargs,
        )
        return int(res["result"] == "true")
