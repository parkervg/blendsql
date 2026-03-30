from blendsql.models.model_base import ModelBase
from blendsql.common.typing import GenerationItem
from .utils import openai_compatible_image_url


class VLLM(ModelBase):
    """Class for vLLM endpoints.

    Args:
        model_name_or_path: Name of the model
        base_url: Base URL for http requests. Defaults to "http://localhost:8000/v1/"

    Examples:
        ```python
        from blendsql.models import VLLM

        model = VLLM("RedHatAI/gemma-3-12b-it-quantized.w4a16", base_url="http://localhost:8000/v1/")
        ```
    """

    def __init__(
        self, api_key: str | None = None, base_url: str | None = None, *args, **kwargs
    ) -> None:
        api_key = api_key or "N.A"
        base_url = base_url or "http://localhost:8000/v1/"
        super().__init__(api_key=api_key, base_url=base_url, *args, **kwargs)

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

        if item.grammar:
            extra_body |= {
                "guided_decoding_backend": "guidance",
                "guided_grammar": item.grammar,
                "structured_outputs": {"grammar": item.grammar},
            }
        return messages, extra_body
