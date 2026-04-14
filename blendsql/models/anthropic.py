import os

from blendsql.models.model_base import ModelBase
from blendsql.common.typing import GenerationItem
from .utils import openai_compatible_image_url


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
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        super().__init__(
            model_name_or_path=model_name_or_path,
            api_key=api_key,
            base_url="https://api.anthropic.com/v1/",
            *args,
            **kwargs,
        )

    async def _format_inputs(
        self, extra_body: dict, item: GenerationItem
    ) -> tuple[list[dict], dict]:
        if len(item.image_urls) > 0:
            content = [{"type": "text", "text": item.prompt}]
            for image_url in item.image_urls:
                session = await self._get_session()
                encoded = await openai_compatible_image_url(image_url, session)
                content.append({"type": "image_url", "image_url": {"url": encoded}})
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": item.prompt}]
        return messages, extra_body
