import importlib.util
from typing import Optional
from functools import cached_property

from blendsql.models.model import UnconstrainedModel, ModelObj

DEFAULT_KWARGS = {"do_sample": False}

_has_transformers = importlib.util.find_spec("transformers") is not None
_has_torch = importlib.util.find_spec("torch") is not None


class TransformersVisionModel(UnconstrainedModel):
    """Wrapper for the image-to-text Transformers pipeline."""

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

    def _load_model(self):
        from transformers import pipeline

        return pipeline(
            "image-to-text",
            model=self.model_name_or_path,
            device_map=self.config.pop("device_map", None),
        )
