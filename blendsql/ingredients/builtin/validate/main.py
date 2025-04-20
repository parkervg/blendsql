from typing import Dict, Union, Optional
import pandas as pd

from blendsql.models import Model, ConstrainedModel
from blendsql.models.constrained.utils import LMString, maybe_load_lm
from blendsql.models.utils import user
from blendsql.ingredients.ingredient import QAIngredient
from blendsql.common.exceptions import IngredientException


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

        serialized_db = context.to_string() if context is not None else ""
        instruction_str = "You are a database expert in charge of validating a claim given a context. Given a claim and associated database context, you will respond 'true' if the claim is factual given the context, and 'false' if not."
        instruction_str += f"Claim: {question}"
        if kwargs.get("table_title"):
            instruction_str += f"\nTable Description: {kwargs.get('table_title')}"
        instruction_str += f"\n{serialized_db}\n\nAnswer:"

        if isinstance(model, ConstrainedModel):
            import guidance

            lm = LMString()
            in_cache = False
            if model.caching:
                response, key = model.check_cache(instruction_str)
                if response is not None:
                    in_cache = True
            if not in_cache:
                # Load our underlying guidance model, if we need to
                lm: guidance.models.Model = maybe_load_lm(model, lm)
                model.num_generation_calls += 1
                with guidance.user():
                    lm += instruction_str

                model.prompt_tokens += len(model.tokenizer.encode(lm._current_prompt()))

                with guidance.assistant():
                    lm += guidance.capture(
                        guidance.select(options=["true", "false"]), name="response"
                    )

                response = lm["response"]
                model.completion_tokens += len(model.tokenizer.encode(str(response)))

                if model.caching:
                    model.cache[key] = response
        else:
            response = model.generate(
                messages_list=[[user(instruction_str)]], max_tokens=5
            )

        # Post-process language model response
        return bool(response == "true")
