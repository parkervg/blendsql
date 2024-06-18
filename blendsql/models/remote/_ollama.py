import importlib.util
from functools import partial

from .._model import RemoteModel
from typing import Optional

_has_ollama = importlib.util.find_spec("ollama") is not None


class OllamaLLM(RemoteModel):
    """Class for an ollama model.

    Args:
        model_name_or_path: Name of the ollama model to load.
            See https://ollama.com/library
        host: Optional custom host to connect to
            e.g. 'http://localhost:11434'
        caching: Bool determining whether we access the model's cache

    Examples:
        ```python
        from blendsql.models import OllamaLLM
        # First, make sure your ollama server is running.
        model = OllamaLLM("phi3")
        ```
    """

    def __init__(
        self,
        model_name_or_path: str,
        host: Optional[str] = None,
        caching: bool = True,
        **kwargs
    ):
        if not _has_ollama:
            raise ImportError(
                "Please install ollama with `pip install ollama`!"
            ) from None

        self.client = None
        if host is not None:
            from ollama import Client

            self.client = Client(host=host)

        super().__init__(
            model_name_or_path=model_name_or_path,
            requires_config=False,
            # TODO: how to get ollama tokenizer?
            tokenizer=None,
            caching=caching,
            **kwargs
        )

    def _load_model(self) -> partial:
        import ollama

        return partial(
            ollama.chat if self.client is None else self.client.chat,
            model=self.model_name_or_path,
        )
