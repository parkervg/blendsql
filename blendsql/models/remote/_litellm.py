import os
from typing import Optional, List
from asyncio import Semaphore
import asyncio
from litellm import acompletion

from ..._configure import ASYNC_LIMIT_KEY, DEFAULT_ASYNC_LIMIT
from .._utils import get_tokenizer
from .._model import RemoteModel

DEFAULT_CONFIG = {"temperature": 0.0}


class LiteLLM(RemoteModel):
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
        config: Optional[dict] = None,
        caching: bool = True,
        **kwargs,
    ):
        if config is None:
            config = {}
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=get_tokenizer(model_name_or_path),
            requires_config=False if model_name_or_path.startswith("ollama") else True,
            load_model_kwargs=config | DEFAULT_CONFIG,
            env=env,
            caching=caching,
            **kwargs,
        )

    async def _generate(
        self,
        messages_list: List[List[dict]],
        max_tokens: Optional[int] = None,
        stop_at: Optional[List[str]] = None,
        **kwargs,
    ):
        sem = Semaphore(int(os.getenv(ASYNC_LIMIT_KEY, DEFAULT_ASYNC_LIMIT)))
        async with sem:
            responses = [
                acompletion(
                    model=self.model_name_or_path,
                    messages=messages,
                    max_tokens=max_tokens,
                    stop=stop_at,
                    **self.load_model_kwargs,
                )
                for messages in messages_list
            ]
        return [m.choices[0].message.content for m in await asyncio.gather(*responses)]

    def generate(self, *args, **kwargs) -> List[str]:
        return asyncio.get_event_loop().run_until_complete(
            self._generate(*args, **kwargs)
        )
