from blendsql.models.model_base import ModelBase
from blendsql.common.typing import GenerationItem
from .utils import openai_compatible_image_url


class Ollama(ModelBase):
    """Class for Ollama endpoints.

    Ollama exposes an OpenAI-compatible API, so no special client is needed.
    Start a local Ollama server with ``ollama serve`` before using this class.

    Args:
        model_name_or_path: Name of the model (e.g. "llama3.2", "mistral", "llava")
        base_url: Base URL for the Ollama server. Defaults to "http://localhost:11434/v1/"
        api_key: API key (Ollama doesn't require one; any value works)

    Examples:
        ```python
        from blendsql.models import Ollama

        model = Ollama("llama3.2")
        model = Ollama("llava", base_url="http://localhost:11434/v1/")
        ```
    """

    def __init__(
        self,
        model_name_or_path: str,
        api_key: str | None = None,
        base_url: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        api_key = api_key or "ollama"
        base_url = base_url or "http://localhost:11434/v1/"
        super().__init__(
            model_name_or_path=model_name_or_path,
            api_key=api_key,
            base_url=base_url,
            *args,
            **kwargs,
        )

    async def _format_inputs(
        self, extra_body: dict, item: GenerationItem
    ) -> tuple[list[dict], dict]:
        if item.audio_urls:
            raise ValueError("Ollama does not support audio inputs.")

        if item.image_urls:
            session = await self._get_session()
            content = []
            for image_url in item.image_urls:
                encoded = await openai_compatible_image_url(image_url, session)
                content.append({"type": "image_url", "image_url": {"url": encoded}})
            content.append({"type": "text", "text": item.prompt})
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": item.prompt}]

        return messages, extra_body
