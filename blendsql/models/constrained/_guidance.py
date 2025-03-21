import importlib.util
from typing import Optional
from colorama import Fore
from functools import cached_property

from ..._logger import logger
from .._model import ConstrainedModel, ModelObj

DEFAULT_KWARGS = {"do_sample": False, "temperature": 0.0, "top_p": 1.0}

_has_transformers = importlib.util.find_spec("transformers") is not None
_has_torch = importlib.util.find_spec("torch") is not None


def resolve_device_map(device_map: Optional[str] = None):
    # cuda -> mps -> cpu
    import torch

    if device_map is not None:
        return device_map

    if torch.cuda.is_available():
        return "cuda"
    else:
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if has_mps:
            return "mps"
        else:
            return "cpu"


class TransformersLLM(ConstrainedModel):
    """Class for Transformers local Model.

    Args:
        model_name_or_path: Name of the model on HuggingFace, or the path to a local model
        caching: Bool determining whether we access the model's cache

    Examples:
        ```python
        from blendsql.models import TransformersLLM
        model = TransformersLLM("Qwen/Qwen1.5-0.5B")
        ```
    """

    def __init__(
        self,
        model_name_or_path: str,
        config: Optional[dict] = None,
        caching: bool = True,
        **kwargs,
    ):
        if not _has_transformers and _has_torch:
            raise ImportError(
                "Please install transformers with `pip install transformers==4.47.0`!"
            ) from None
        elif not _has_torch and _has_transformers:
            raise ImportError(
                "Please install pytorch with `pip install torch`!"
            ) from None
        elif not _has_torch and not _has_transformers:
            raise ImportError(
                "Please install transformers and pytorch with `pip install transformers==4.47.0 torch`!"
            ) from None
        import transformers

        transformers.logging.set_verbosity_error()
        if config is None:
            config = {}

        super().__init__(
            model_name_or_path=model_name_or_path,
            requires_config=False,
            tokenizer=transformers.AutoTokenizer.from_pretrained(model_name_or_path),
            config=DEFAULT_KWARGS | config,
            caching=caching,
            **kwargs,
        )

    @cached_property
    def model_obj(self) -> ModelObj:
        """Allows for lazy loading of underlying model weights."""
        return self._load_model()

    def _load_model(self) -> ModelObj:
        # https://huggingface.co/blog/how-to-generate
        from guidance.models import Transformers

        device_map = resolve_device_map(self.config.pop("device_map", None))

        lm = Transformers(
            self.model_name_or_path,
            echo=False,
            device_map=device_map,
            **self.config,
        )
        # Try to infer if we're in chat mode
        if lm.engine.tokenizer._orig_tokenizer.chat_template is None:
            logger.debug(
                Fore.YELLOW
                + "chat_template not found in tokenizer config.\nBlendSQL currently only works with chat models"
                + Fore.RESET
            )
        return lm


class TransformersVisionModel(TransformersLLM):
    """Wrapper for the image-to-text Transformers pipeline."""

    def _load_model(self):
        from transformers import pipeline

        device_map = resolve_device_map(self.config.pop("device_map", None))

        return pipeline(
            "image-to-text", model=self.model_name_or_path, device_map=device_map
        )
