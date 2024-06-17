import importlib.util
from outlines.models.llamacpp import llamacpp

from .._model import LocalModel, ModelObj
from typing import Optional

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
        self,
        model_name_or_path: str,
        filename: str,
        hf_repo_with_config: Optional[str] = None,
        caching: bool = True,
        **kwargs
    ):
        if not _has_llama_cpp:
            raise ImportError(
                "Please install llama_cpp with `pip install llama-cpp-python`!"
            ) from None
        from llama_cpp import llama_tokenizer

        self._llama_tokenizer = None
        if hf_repo_with_config:
            self._llama_tokenizer = llama_tokenizer.LlamaHFTokenizer.from_pretrained(
                hf_repo_with_config
            )

        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=self._llama_tokenizer.hf_tokenizer
            if self._llama_tokenizer is not None
            else None,
            requires_config=False,
            load_model_kwargs=kwargs | {"filename": filename},
            caching=caching,
        )

    def _load_model(self, filename: str) -> ModelObj:
        return llamacpp(
            self.model_name_or_path,
            filename=filename,
            tokenizer=self._llama_tokenizer,
            **self.load_model_kwargs
        )
