from blendsql.models.model_base import ModelBase
from blendsql.common.typing import GenerationItem


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

    def _format_extra_body(self, extra_body: dict, item: GenerationItem) -> dict:
        return extra_body
