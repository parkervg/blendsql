from typing import Dict, Union, Optional, Tuple
import pandas as pd
import guidance

from blendsql.models import Model, LocalModel
from blendsql._program import Program
from blendsql.ingredients.ingredient import QAIngredient
from blendsql._exceptions import IngredientException
from blendsql.ingredients.generate import generate


class ValidateProgram(Program):
    def __call__(
        self,
        model: Model,
        question: str,
        context: Optional[pd.DataFrame] = None,
        table_title: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, str]:
        m: guidance.models.Model = model.model_obj
        serialized_db = context.to_string() if context is not None else ""
        m += "You are a database expert in charge of validating a claim given a context. Given a claim and associated database context, you will respond 'true' if the claim is factual given the context, and 'false' if not."
        m += f"Claim: {question}"
        if table_title:
            m += f"\nTable Description: {table_title}"
        m += f"\n{serialized_db}\n\nAnswer:"
        prompt = m._current_prompt()
        if isinstance(model, LocalModel):
            response = (
                m
                + guidance.capture(
                    guidance.select(options=["true", "false"]), name="result"
                )["result"]
            )
        else:
            response = generate(model, prompt=prompt, max_tokens=5)
        return (response, prompt)


class LLMValidate(QAIngredient):
    def run(
        self,
        model: Model,
        question: str,
        context: Optional[pd.DataFrame] = None,
        value_limit: Optional[int] = None,
        table_to_title: Optional[Dict[str, str]] = None,
        long_answer: bool = False,
        **kwargs,
    ) -> Union[str, int, float]:
        if model is None:
            raise IngredientException(
                "LLMValidate requires a `Model` object, but nothing was passed!\nMost likely you forgot to set the `default_model` argument in `blend()`"
            )
        if context is not None:
            if value_limit is not None:
                context = context.iloc[:value_limit]
        response = model.predict(
            program=ValidateProgram,
            question=question,
            context=context,
            table_title=None,
            **kwargs,
        )
        # Post-process language model response
        return bool(response == "true")
