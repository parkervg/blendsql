import os
import asyncio

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
        config: dict | None = None,
        caching: bool = False,
        **kwargs,
    ):
        raise NotImplementedError(
            "`UnconstrainedModel` classes (LiteLLM endpoints like OpenAI, Anthropic) are currently not functioning.\nUse a local ConstrainedModel (llama.cpp, transformers) instead! Contributions to the UnconstrainedModel class are welcome."
        )
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
        messages_list: list[list[dict]],
        max_tokens: int | None = None,
        stop_at: list[str] | None = None,
        **kwargs,
    ):
        from litellm import acompletion
        from asyncio import Semaphore

        sem = Semaphore(int(os.getenv(ASYNC_LIMIT_KEY, DEFAULT_ASYNC_LIMIT)))
        async with sem:
            responses = [
                acompletion(
                    model=self.model_name_or_path,
                    messages=messages,
                    max_tokens=max_tokens,
                    stop=stop_at,
                    **self.config,
                )
                for messages in messages_list
            ]
            return [m for m in await asyncio.gather(*responses)]

    def generate(self, *args, **kwargs) -> list[str]:
        """Handles cache lookup and generation using LiteLLM."""
        responses, key = None, None
        if self.caching:
            responses, key = self.check_cache(*args, **kwargs)
        if responses is None:
            asyncio.set_event_loop(asyncio.new_event_loop())
            responses = asyncio.get_event_loop().run_until_complete(
                self._generate(*args, **kwargs)
            )  # type: ignore
        self.prompt_tokens += sum([r.usage.prompt_tokens for r in responses])
        self.completion_tokens += sum([r.usage.completion_tokens for r in responses])
        if self.caching:
            self.cache[key] = responses
        return [r.choices[0].message.content for r in responses]
