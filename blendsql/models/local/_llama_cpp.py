import importlib.util
from outlines.models import llamacpp, LogitsGenerator

from .._model import LocalModel

_has_llama_cpp = importlib.util.find_spec("llama_cpp") is not None


class LlamaCppLLM(LocalModel):
    """Class for llama-cpp local Model.

    Args:
        model_name_or_path: Name of the model on HuggingFace, or the path to a local model
    """

    def __init__(
        self, model_name_or_path: str, filename: str, caching: bool = True, **kwargs
    ):
        if not _has_llama_cpp:
            raise ImportError(
                "Please install llama_cpp with `pip install llama-cpp-python`!"
            ) from None

        super().__init__(
            model_name_or_path=model_name_or_path,
            # TODO: how to get llama_cpp tokenizer?
            tokenizer=None,
            requires_config=False,
            load_model_kwargs=kwargs | {"filename": filename},
            caching=caching,
        )

    def _load_model(self, filename: str, **kwargs) -> LogitsGenerator:
        return llamacpp(self.model_name_or_path, filename=filename, **kwargs)
