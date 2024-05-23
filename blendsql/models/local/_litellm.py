import importlib.util
from guidance.models import LiteLLMChat
from guidance.models import Model as GuidanceModel

from .._model import Model

_has_litellm = importlib.util.find_spec("litellm") is not None


# class GrammarlessGuidanceModel(list):
#     def __init__(self, model_name_or_path: str):
#         self.model_name_or_path = model_name_or_path
#         self._variables = {}
#
#     def _current_prompt(self):
#         return "".join(self)


class LiteLLM(Model):
    def __init__(self, model_name_or_path: str, caching: bool = True, **kwargs):
        if not _has_litellm:
            raise ImportError(
                "Please install litellm with `pip install litellm`!"
            ) from None

        super().__init__(
            model_name_or_path=model_name_or_path,
            requires_config=False,
            # TODO: how to get ollama tokenizer?
            tokenizer=None,
            caching=caching,
            **kwargs
        )

    def _load_model(self) -> GuidanceModel:
        return LiteLLMChat(self.model_name_or_path, echo=False)
