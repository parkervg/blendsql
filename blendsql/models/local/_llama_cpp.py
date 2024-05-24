import importlib.util
from outlines.models import llamacpp, LogitsGenerator

from .._model import LocalModel

_has_llama_cpp = importlib.util.find_spec("llama_cpp") is not None


class LlamaCppLLM(LocalModel):
    """Class for llama-cpp local Model.

    Args:
        model_name_or_path: Name of the model on HuggingFace
        filename: The specific .gguf file in the HuggingFace repo to load
        caching: Bool determining whether we access the model's cache

    Examples:
        ```python
        from blendsql.models import LlamaCppLLM
        model = LlamaCppLLM(
            "TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF",
            filename="tinyllama-1.1b-1t-openorca.Q2_K.gguf"
        )
        ```
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
