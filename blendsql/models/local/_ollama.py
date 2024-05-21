import logging
import importlib.util


from .._model import Model

logging.getLogger("guidance").setLevel(logging.CRITICAL)

_has_ollama = importlib.util.find_spec("ollama") is not None


class OllamaGuidanceModel(list):
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self._variables = {}

    def _current_prompt(self):
        return "".join(self)


class OllamaLLM(Model):
    def __init__(self, model_name_or_path: str, caching: bool = True, **kwargs):
        if not _has_ollama:
            raise ImportError(
                "Please install ollama with `pip install ollama`!"
            ) from None

        super().__init__(
            model_name_or_path=model_name_or_path,
            requires_config=False,
            # TODO: how to get ollama tokenizer?
            tokenizer=None,
            caching=caching,
            **kwargs
        )

    def _load_model(self) -> OllamaGuidanceModel:
        return OllamaGuidanceModel(self.model_name_or_path)
