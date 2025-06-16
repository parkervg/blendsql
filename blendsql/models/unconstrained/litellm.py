import os
import typing as t
from asyncio import Semaphore
import asyncio
import dspy
from colorama import Fore

from blendsql.common.logger import logger
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
        kwargs_list: t.List[dict],
        max_tokens: t.Optional[int] = None,
        stop: t.Optional[t.List[str]] = None,
    ):
        sem = Semaphore(int(os.getenv(ASYNC_LIMIT_KEY, DEFAULT_ASYNC_LIMIT)))
        dspy.configure(
            lm=dspy.LM(
                self.model_name_or_path,
                cache=False,
                **{**{"max_tokens": max_tokens, "stop": stop}, **self.config},
            )
        )

        async with sem:
            responses = [dspy_predict.acall(**kwargs) for kwargs in kwargs_list]
            return [m for m in await asyncio.gather(*responses)]

    def generate(
        self,
        dspy_predict: dspy.Predict,
        kwargs_list: t.List[dict],
        *args,
        **kwargs,
    ) -> t.List[t.Any]:
        """Handles cache lookup and generation using LiteLLM.

        Returns responses in the same order as the input kwargs_list, using cache where available
        and generating new responses for cache misses.
        """
        return asyncio.get_event_loop().run_until_complete(
            self._generate(dspy_predict, kwargs_list, *args, **kwargs)
        )

    def get_usage(self, n: int) -> dict:
        return [i["usage"] for i in dspy.settings.lm.history[-n:]]

    def get_token_usage(self, n: int) -> t.Tuple[int, int]:
        prompt_tokens, completion_tokens = 0, 0
        usages = self.get_usage(n)
        for usage in usages:
            if usage != {}:
                prompt_tokens += usage["prompt_tokens"]
                completion_tokens += usage["completion_tokens"]
            else:
                logger.debug(
                    Fore.RED
                    + "DSPy program has empty usage. Is caching on by accident?"
                    + Fore.RESET
                )
        return (prompt_tokens, completion_tokens)
