import os

from blendsql.models.model_base import ModelBase
from blendsql.common.typing import GenerationItem
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
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        super().__init__(
            model_name_or_path=model_name_or_path,
            api_key=api_key,
            base_url="https://api.anthropic.com/v1/",
            *args,
            **kwargs,
        )

    async def _format_extra_body(
        self, extra_body: dict, item: GenerationItem
    ) -> tuple[list[dict], dict]:
        if item.image_url is not None:
            session = await self._get_session()
            base64_image = await get_base64_string(item.image_url, session)
            filetype = item.image_url.split(".")[-1].lower()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": item.prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{filetype}",
                                "data": base64_image,
                            },
                        },
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": item.prompt}]
        return messages, extra_body
