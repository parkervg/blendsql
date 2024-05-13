import logging
import importlib.util
from guidance.models import Transformers, Model

from .._model import Model

logging.getLogger("guidance").setLevel(logging.CRITICAL)

_has_transformers = importlib.util.find_spec("transformers") is not None


class TransformersLLM(Model):
    """Class for Transformers local Model.

    Args:
        model_name_or_path: Name of the model on HuggingFace, or the path to a local model
    """

    def __init__(self, model_name_or_path: str, **kwargs):
        if not _has_transformers:
            raise ImportError(
                "Please install transformers with `pip install transformers`!"
            ) from None
        import transformers

        super().__init__(
            model_name_or_path=model_name_or_path,
            requires_config=False,
            tokenizer=transformers.AutoTokenizer.from_pretrained(model_name_or_path),
            **kwargs
        )

    def _load_model(self) -> Model:
        return Transformers(self.model_name_or_path, echo=False)
