import os

from blendsql.models.model_base import ModelBase
from blendsql.common.typing import GenerationItem
from .utils import openai_compatible_image_url


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

    async def _format_extra_body(
        self, extra_body: dict, item: GenerationItem
    ) -> tuple[list[dict], dict]:
        if item.image_url is not None:
            session = await self._get_session()
            encoded = await openai_compatible_image_url(item.image_url, session)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": item.prompt},
                        {"type": "image_url", "image_url": encoded},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": item.prompt}]

        if "max_tokens" in extra_body:
            extra_body["max_completion_tokens"] = extra_body.pop("max_tokens")
        return messages, extra_body
