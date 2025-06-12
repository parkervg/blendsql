import os
import typing as t
from asyncio import Semaphore
import asyncio
import dspy

from blendsql.configure import ASYNC_LIMIT_KEY, DEFAULT_ASYNC_LIMIT
from blendsql.models.model import UnconstrainedModel

DEFAULT_CONFIG = {"temperature": 0.0}


class LiteLLM(UnconstrainedModel):
    """Class for LiteLLM remote model integration.
    https://github.com/BerriAI/litellm

    Args:
        model_name_or_path: Name or identifier of the model to use with LiteLLM.
            Should begin with provider, e.g. `openai/gpt-3.5-turbo`, `gemini/gemini-2.0-flash-exp`, `anthropic/claude-3-7-sonnet-20250219`.
        env: Environment path, defaults to current directory (".")
        config: Optional dictionary containing model configuration parameters
        caching: Bool determining whether to enable response caching
        **kwargs: Additional keyword arguments to pass to the model

    Examples:
        ```python
        from blendsql.models import LiteLLM
        model = LiteLLM("openai/gpt-4o-mini", config={"temperature": 0.7})
        ```
    """

    def __init__(
        self,
        model_name_or_path: str,
        env: str = ".",
        config: t.Optional[dict] = None,
        caching: bool = True,
        **kwargs,
    ):
        if config is None:
            config = {}
        super().__init__(
            model_name_or_path=model_name_or_path,
            requires_config=False if model_name_or_path.startswith("ollama") else True,
            config=DEFAULT_CONFIG | config,
            env=env,
            caching=caching,
            **kwargs,
        )
        self.model_obj = None

    async def _generate(
        self,
        dspy_predict: dspy.Predict,
        kwargs_list: t.List[t.List[dict]],
        max_tokens: t.Optional[int] = None,
        stop: t.Optional[t.List[str]] = None,
    ):
        sem = Semaphore(int(os.getenv(ASYNC_LIMIT_KEY, DEFAULT_ASYNC_LIMIT)))
        dspy.configure(
            lm=dspy.LM(
                self.model_name_or_path,
                {**{"max_tokens": max_tokens, "stop": stop}, **self.config},
            )
        )

        async with sem:
            responses = [
                dspy_predict.acall(model=self.model_name_or_path, **kwargs)
                for kwargs in kwargs_list
            ]
            return [m for m in await asyncio.gather(*responses)]

    def generate(
        self,
        dspy_predict: dspy.Predict,
        kwargs_list: t.List[t.List[dict]],
        *args,
        **kwargs,
    ) -> t.List[str]:
        """Handles cache lookup and generation using LiteLLM."""
        responses = asyncio.get_event_loop().run_until_complete(
            self._generate(dspy_predict, kwargs_list, *args, **kwargs)
        )  # type: ignore
        for call in dspy.settings.lm.history[-len(kwargs_list) :]:
            self.num_generation_calls += bool(call["usage"] != {})
            if call["usage"] != dict():
                self.prompt_tokens += call["usage"]["prompt_tokens"]
                self.completion_tokens += call["usage"]["completion_tokens"]
        return [r.answer for r in responses]
