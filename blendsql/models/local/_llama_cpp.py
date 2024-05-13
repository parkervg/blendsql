import logging
import importlib.util
from guidance.models import LlamaCpp

from .._model import Model

logging.getLogger("guidance").setLevel(logging.CRITICAL)

_has_llama_cpp = importlib.util.find_spec("llama_cpp") is not None


class LlamaCppLLM(Model):
    """Class for Transformers local Model.

    Args:
        model_name_or_path: Name of the model on HuggingFace, or the path to a local model
    """

    def __init__(self, model_name_or_path: str, **kwargs):
        if not _has_llama_cpp:
            raise ImportError(
                "Please install llama_cpp with `pip install llama_cpp`!"
            ) from None

        super().__init__(
            model_name_or_path=model_name_or_path,
            requires_config=False,
            # TODO: how to get llama_cpp tokenizer?
            tokenizer=None,
            **kwargs
        )

    def _load_model(self) -> Model:
        return LlamaCpp(self.model_name_or_path, n_ctx=2048, echo=False)
