from typing import Dict, Union, Optional, Tuple
import pandas as pd
import outlines

from blendsql.models import Model
from blendsql._program import Program
from blendsql.ingredients.ingredient import QAIngredient


class ValidateProgram(Program):
    def __call__(
        self,
        question: str,
        context: pd.DataFrame = None,
        table_title: str = None,
        **kwargs,
    ) -> Tuple[str, str]:
        serialized_db = context.to_string() if context is not None else ""
        prompt = ""
        prompt += "You are a database expert in charge of validating a claim given a context. Given a claim and associated database context, you will respond 'true' if the claim is factual given the context, and 'false' if not."
        prompt += f"Claim: {question}"
        if table_title:
            prompt += f"\nTable Description: {table_title}"
        prompt += f"\n{serialized_db}\n\nAnswer:"
        generator = outlines.generate.choice(
            self.model.logits_generator, ["true", "false"]
        )
        return (generator(prompt), prompt)


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
        response = model.predict(
            program=ValidateProgram,
            question=question,
            context=context,
            table_title=None,
            **kwargs,
        )
        return int(response == "true")
