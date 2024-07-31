import importlib.util
from .._model import LocalModel, ModelObj

DEFAULT_KWARGS = {"do_sample": True, "temperature": 0.0, "top_p": 1.0}

_has_transformers = importlib.util.find_spec("transformers") is not None
_has_torch = importlib.util.find_spec("torch") is not None


class TransformersLLM(LocalModel):
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

    def __init__(self, model_name_or_path: str, caching: bool = True, **kwargs):
        if not _has_transformers and _has_torch:
            raise ImportError(
                "Please install transformers with `pip install transformers`!"
            ) from None
        elif not _has_torch and _has_transformers:
            raise ImportError(
                "Please install pytorch with `pip install torch`!"
            ) from None
        elif not _has_torch and not _has_transformers:
            raise ImportError(
                "Please install transformers and pytorch with `pip install transformers torch`!"
            ) from None
        import transformers

        transformers.logging.set_verbosity_error()

        super().__init__(
            model_name_or_path=model_name_or_path,
            requires_config=False,
            tokenizer=transformers.AutoTokenizer.from_pretrained(model_name_or_path),
            load_model_kwargs=DEFAULT_KWARGS | kwargs,
            caching=caching,
            **kwargs,
        )

    def _load_model(self) -> ModelObj:
        # https://huggingface.co/blog/how-to-generate
        from outlines.models import transformers

        return transformers(
            self.model_name_or_path,
            model_kwargs=self.load_model_kwargs,
        )


class TransformersVisionModel(TransformersLLM):
    """Wrapper for the image-to-text Transformers pipeline."""

    def __init__(
        self, model_name_or_path: str, model_class, caching: bool = True, **kwargs
    ):
        self.model_class = model_class
        if not _has_transformers:
            raise ImportError(
                "Please install transformers with `pip install transformers`!"
            ) from None
        else:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        super().__init__(
            model_name_or_path=model_name_or_path,
            caching=caching,
            **kwargs,
        )

    def _load_model(self):
        from outlines.models import transformers_vision

        return transformers_vision(
            model_name=self.model_name_or_path, model_class=self.model_class
        )
