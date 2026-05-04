import os

from blendsql.models.model_base import ModelBase
from blendsql.common.typing import GenerationItem
from .utils import openai_compatible_image_url, openai_compatible_audio_url


class OpenAI(ModelBase):
    """Class for OpenAI endpoints.

    Args:
        model_name_or_path: Name of the model
        api_key (optional): API key for OpenAI API. Include this, or set the `OPENAI_API_KEY` environment variable.

    Examples:
        ```python
        from blendsql.models import OpenAI

        model = OpenAI("gpt-5")
        ```
    """

    def __init__(
        self, model_name_or_path: str, api_key: str | None = None, *args, **kwargs
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        super().__init__(
            model_name_or_path=model_name_or_path, api_key=api_key, *args, **kwargs
        )

    async def _format_inputs(
        self, extra_body: dict, item: GenerationItem
    ) -> tuple[list[dict], dict]:
        if item.image_urls or item.audio_urls:
            session = await self._get_session()
            content = [{"type": "text", "text": item.prompt}]
            for image_url in item.image_urls:
                encoded = await openai_compatible_image_url(image_url, session)
                content.append({"type": "image_url", "image_url": {"url": encoded}})
            for audio_url in item.audio_urls:
                audio_data = await openai_compatible_audio_url(audio_url, session)
                content.append({"type": "input_audio", "input_audio": audio_data})
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": item.prompt}]

        if "max_tokens" in extra_body:
            extra_body["max_completion_tokens"] = extra_body.pop("max_tokens")
        return messages, extra_body
