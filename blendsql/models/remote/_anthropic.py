import os
import importlib.util
from typing import Optional

from .._model import RemoteModel, ModelObj

DEFAULT_CONFIG = {"temperature": 0.0}

_has_anthropic = importlib.util.find_spec("anthropic") is not None


class AnthropicLLM(RemoteModel):
    """Class for Anthropic Model API.

    Args:
        model_name_or_path: Name of the Anthropic model to use
        env: Path to directory of .env file, or to the file itself to load as a dotfile.
            Should contain the variable `ANTHROPIC_API_KEY`
        config: Optional argument mapping to use in loading model
        caching: Bool determining whether we access the model's cache

    Examples:
        Given the following `.env` file in the directory above current:
        ```text
        ANTHROPIC_API_KEY=my_api_key
        ```
        ```python
        from blendsql.models import AnthropicLLM

        model = AnthropicLLM(
            "claude-3-5-sonnet-20240620",
            env="..",
            config={"temperature": 0.7}
        )
        ```
    """

    def __init__(
        self,
        model_name_or_path: str,
        env: str = ".",
        config: Optional[dict] = None,
        caching: bool = True,
        **kwargs,
    ):
        if not _has_anthropic:
            raise ImportError(
                "Please install anthropic with `pip install anthropic`!"
            ) from None

        if config is None:
            config = {}
        super().__init__(
            model_name_or_path=model_name_or_path,
            # No public anthropic tokenizer ):
            tokenizer=None,
            requires_config=True,
            refresh_interval_min=30,
            load_model_kwargs=config | DEFAULT_CONFIG,
            env=env,
            caching=caching,
            **kwargs,
        )

    def _load_model(self) -> ModelObj:
        from guidance.models import Anthropic

        return Anthropic(
            self.model_name_or_path, echo=False, api_key=os.getenv("ANTHROPIC_API_KEY")
        )
