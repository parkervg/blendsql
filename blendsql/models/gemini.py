import os

from blendsql.models.model_base import ModelBase
from blendsql.common.typing import GenerationItem
from .utils import openai_compatible_image_url, openai_compatible_audio_url

_GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class Gemini(ModelBase):
    """Class for Google Gemini endpoints via the OpenAI-compatible API.

    No extra SDK is required beyond the ``openai`` package already used by
    blendsql.  Set the ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) environment
    variable, or pass ``api_key`` directly.

    Args:
        model_name_or_path: Name of the model (e.g. "gemini-2.0-flash", "gemini-2.5-pro")
        api_key (optional): Google AI Studio API key. Include this, or set the
            ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) environment variable.

    Examples:
        ```python
        from blendsql.models import Gemini

        model = Gemini("gemini-2.0-flash")
        model = Gemini("gemini-2.5-pro", api_key="AIza...")
        ```
    """

    def __init__(
        self,
        model_name_or_path: str,
        api_key: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        super().__init__(
            model_name_or_path=model_name_or_path,
            api_key=api_key,
            base_url=_GEMINI_OPENAI_BASE_URL,
            *args,
            **kwargs,
        )

    async def _format_inputs(
        self, extra_body: dict, item: GenerationItem
    ) -> tuple[list[dict], dict]:
        if item.image_urls or item.audio_urls:
            session = await self._get_session()
            content = []
            for image_url in item.image_urls:
                encoded = await openai_compatible_image_url(image_url, session)
                content.append({"type": "image_url", "image_url": {"url": encoded}})
            for audio_url in item.audio_urls:
                audio_data = await openai_compatible_audio_url(audio_url, session)
                content.append({"type": "input_audio", "input_audio": audio_data})
            content.append({"type": "text", "text": item.prompt})
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": item.prompt}]

        return messages, extra_body
