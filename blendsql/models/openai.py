from blendsql.models.model_base import ModelBase
from blendsql.common.typing import GenerationItem


class OpenAI(ModelBase):
    """Class for vLLM endpoints.

    Args:
        model_name_or_path: Name of the model
        api_key (optional): API key for OpenAI API. Include this, or set the `OPENAI_API_KEY` environment variable.

    Examples:
        ```python
        from blendsql.models import VLLM

        model = OpenAI("gpt-5")
        ```
    """

    async def _format_extra_body(self, extra_body: dict, item: GenerationItem) -> dict:
        return extra_body
