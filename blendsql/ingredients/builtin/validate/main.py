from typing import Dict, Union, Optional
import pandas as pd
import guidance

from blendsql.models import Model, ConstrainedModel
from blendsql.models._utils import user
from blendsql.ingredients.ingredient import QAIngredient
from blendsql._exceptions import IngredientException


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
        m: guidance.models.Model = model.model_obj
        serialized_db = context.to_string() if context is not None else ""
        m += "You are a database expert in charge of validating a claim given a context. Given a claim and associated database context, you will respond 'true' if the claim is factual given the context, and 'false' if not."
        m += f"Claim: {question}"
        if kwargs.get("table_title"):
            m += f"\nTable Description: {kwargs.get('table_title')}"
        m += f"\n{serialized_db}\n\nAnswer:"
        prompt = m._current_prompt()
        if isinstance(model, ConstrainedModel):
            response = (
                m
                + guidance.capture(
                    guidance.select(options=["true", "false"]), name="result"
                )["result"]
            )
        else:
            response = model.generate(messages_list=[[user(prompt)]], max_tokens=5)
        # Post-process language model response
        return bool(response == "true")
