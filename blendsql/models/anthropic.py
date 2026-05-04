import os
import asyncio

from blendsql.models.model_base import ModelBase
from blendsql.common.typing import GenerationItem, GenerationResult
from blendsql.configure import MAX_TOKENS_KEY, DEFAULT_MAX_TOKENS, add_to_global_history
from .utils import get_base64_string


class Anthropic(ModelBase):
    """Class for Anthropic endpoints.

    Args:
        model_name_or_path: Name of the model
        api_key (optional): API key for Anthropic API. Include this, or set the `ANTHROPIC_API_KEY` environment variable.

    Examples:
        ```python
        from blendsql.models import Anthropic

        model = Anthropic("claude-opus-4-6")
        ```
    """

    def __init__(
        self, model_name_or_path: str, api_key: str | None = None, *args, **kwargs
    ):
        import anthropic as anthropic_sdk

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        super().__init__(
            model_name_or_path=model_name_or_path,
            api_key=api_key,
            *args,
            **kwargs,
        )
        self.client = anthropic_sdk.AsyncAnthropic(api_key=api_key)

    async def _format_inputs(
        self, extra_body: dict, item: GenerationItem
    ) -> tuple[list[dict], dict]:
        session = await self._get_session()
        content = []

        for image_url in item.image_urls:
            if image_url.startswith(("http://", "https://")):
                content.append(
                    {"type": "image", "source": {"type": "url", "url": image_url}}
                )
            else:
                filetype = image_url.split(".")[-1].lower()
                if filetype == "jpg":
                    filetype = "jpeg"
                base64_data = await get_base64_string(image_url, session)
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{filetype}",
                            "data": base64_data,
                        },
                    }
                )

        for audio_url in item.audio_urls:
            filetype = audio_url.split(".")[-1].lower()
            base64_data = await get_base64_string(audio_url, session)
            content.append(
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": f"audio/{filetype}",
                        "data": base64_data,
                    },
                }
            )

        content.append({"type": "text", "text": item.prompt})

        messages = [{"role": "user", "content": content}]
        if item.assistant_continuation:
            messages.append(
                {"role": "assistant", "content": item.assistant_continuation}
            )

        return messages, extra_body

    async def generate(
        self, item: GenerationItem, cancel_event: asyncio.Event | None = None
    ) -> GenerationResult:
        max_tokens = int(os.getenv(MAX_TOKENS_KEY, DEFAULT_MAX_TOKENS))
        messages, _ = await self._format_inputs({}, item)

        buffer = ""

        async with self.client.messages.stream(
            model=self.model_name_or_path,
            max_tokens=max_tokens,
            messages=messages,
        ) as stream:
            self.num_generation_calls += 1
            async for text in stream.text_stream:
                if cancel_event and cancel_event.is_set():
                    return GenerationResult(item.identifier, buffer, completed=False)
                buffer += text

            message = await stream.get_final_message()
            self.prompt_tokens += message.usage.input_tokens
            self.completion_tokens += message.usage.output_tokens
            if (
                hasattr(message.usage, "cache_read_input_tokens")
                and message.usage.cache_read_input_tokens
            ):
                self.cached_tokens += message.usage.cache_read_input_tokens

        add_to_global_history(
            f"[USER]{item.prompt}[/USER]\n\n[ASSISTANT]{buffer}[/ASSISTANT]"
        )
        return GenerationResult(item.identifier, buffer, completed=True)
