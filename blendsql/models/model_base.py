from typing import Any, Sequence, Callable, Tuple
import os
from pathlib import Path
from diskcache import Cache
import platformdirs
import hashlib
import inspect
from textwrap import dedent
import aiohttp
import asyncio

from blendsql.common.logger import logger, Color
from blendsql.common.typing import GenerationResult, GenerationItem
from blendsql.configure import MAX_TOKENS_KEY, DEFAULT_MAX_TOKENS, add_to_global_history

CONTEXT_TRUNCATION_LIMIT = 100


class ModelBase:
    """Parent class for all BlendSQL Models."""

    def __init__(
        self,
        model_name_or_path: str,
        base_url: str | None = None,
        api_key: str = "N/A",
        extra_body: dict | None = None,
        chat_template_kwargs: dict | None = None,
        caching: bool = False,
        **kwargs,
    ):
        from openai import AsyncOpenAI

        self.model_name_or_path = model_name_or_path
        self.caching = caching
        self.extra_body = extra_body or dict()
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        if chat_template_kwargs is None:
            self.chat_template_kwargs = {}
        if "chat_template_kwargs" in self.extra_body:
            self.chat_template_kwargs = self.extra_body.pop("chat_template_kwargs")
        self.cache = Cache(
            Path(platformdirs.user_cache_dir("blendsql"))
            / f"{self.model_name_or_path}.diskcache"
        )
        self._session: aiohttp.ClientSession | None = None

        # Initialize counters
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.cached_tokens: int = 0
        self.num_generation_calls: int = 0
        self.num_cache_hits: int = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def generate(
        self, item: GenerationItem, cancel_event: asyncio.Event | None = None
    ):
        buffer = ""
        extra_body = self.extra_body

        messages, extra_body = await self._format_inputs(extra_body, item)

        stream = await self.client.chat.completions.create(
            model=self.model_name_or_path,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra_body,
            max_tokens=int(os.getenv(MAX_TOKENS_KEY, DEFAULT_MAX_TOKENS)),
        )
        self.num_generation_calls += 1

        try:
            async for chunk in stream:
                if cancel_event and cancel_event.is_set():
                    return GenerationResult(item.identifier, buffer, completed=False)

                if chunk.choices and chunk.choices[0].delta.content:
                    buffer += chunk.choices[0].delta.content

                if hasattr(chunk, "usage") and chunk.usage is not None:
                    self.prompt_tokens += chunk.usage.prompt_tokens
                    self.completion_tokens += chunk.usage.completion_tokens
                    if chunk.usage.prompt_tokens_details is not None:
                        self.cached_tokens += (
                            chunk.usage.prompt_tokens_details.cached_tokens
                        )

        finally:
            await stream.close()

        add_to_global_history(
            f"[USER]{item.prompt}[/USER]\n\n[ASSISTANT]{buffer}[/ASSISTANT]"
        )
        return GenerationResult(item.identifier, buffer, completed=True)

    def _create_key(
        self, *args, funcs: Sequence[Callable] | None = None, **kwargs
    ) -> str:
        """Generates a hash to use in diskcache Cache.
        This way, we don't need to send our prompts to the same Model
        if our context of Model + args + kwargs is the same.

        Returns:
            md5 hash used as key in diskcache
        """
        hasher = hashlib.md5()
        params_str = ""
        if len(kwargs) > 0:
            params_str += str(sorted([(k, str(v)) for k, v in kwargs.items()]))
        if len(args) > 0:
            params_str += str([arg for arg in args])
        if funcs:
            params_str += "\n".join([dedent(inspect.getsource(func)) for func in funcs])
        combined_str = "{}||{}".format(
            f"{self.model_name_or_path}||{type(self)}",
            params_str,
        ).encode()
        hasher.update(combined_str)
        return hasher.hexdigest()

    def check_cache(
        self, *args, funcs: Sequence[Callable] | None = None, **kwargs
    ) -> Tuple[Any, str]:
        response: dict[str, str] = None  # type: ignore
        key: str = self._create_key(funcs=funcs, *args, **kwargs)
        if key in self.cache:
            self.num_cache_hits += 1
            logger.debug(
                Color.model_or_data_update(
                    f"Using model cache ({self.num_cache_hits})..."
                )
            )
            response = self.cache.get(key)  # type: ignore
        return (response, key)

    def reset_stats(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.num_generation_calls = 0
        self.num_cache_hits = 0
