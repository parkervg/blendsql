import importlib.util
from typing import Optional
from colorama import Fore
from functools import cached_property

from ..._logger import logger
from .._model import ConstrainedModel, ModelObj

DEFAULT_KWARGS = {"do_sample": False}

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


def infer_chat_template(model_name_or_path: str) -> dict:
    # Try to infer chat template
    chat_template = None
    if "smollm" in model_name_or_path.lower():
        from guidance.chat import ChatMLTemplate

        chat_template = ChatMLTemplate
    elif "llama-3" in model_name_or_path.lower():
        from guidance.chat import Llama3ChatTemplate

        chat_template = Llama3ChatTemplate
    elif "llama-2" in model_name_or_path.lower():
        from guidance.chat import Llama2ChatTemplate

        chat_template = Llama2ChatTemplate
    elif "phi-3" in model_name_or_path.lower():
        from guidance.chat import Phi3MiniChatTemplate

        chat_template = Phi3MiniChatTemplate
    if chat_template is not None:
        logger.debug(
            Fore.MAGENTA
            + f"Loading '{model_name_or_path}' with '{chat_template.__name__}' chat template..."
            + Fore.RESET
        )
    return chat_template


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

        if "chat_template" not in self.config:
            self.config["chat_template"] = infer_chat_template(self.model_name_or_path)

        lm = Transformers(
            self.model_name_or_path,
            echo=False,
            device_map=self.config.pop("device_map", None),
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


class LlamaCpp(ConstrainedModel):
    """Class for LlamaCpp local Model.

    Args:
        model_name_or_path: Name of the model on HuggingFace, or the path to a local model
        caching: Bool determining whether we access the model's cache

    Examples:
        ```python
        from blendsql.models import LlamaCpp
        model = LlamaCpp("Qwen/Qwen1.5-0.5B")
        ```
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
        config: Optional[dict] = None,
        caching: bool = True,
        **kwargs,
    ):
        if config is None:
            config = {}

        super().__init__(
            model_name_or_path=model_name_or_path,
            requires_config=False,
            tokenizer=None,
            config=DEFAULT_KWARGS | config,
            caching=caching,
            **kwargs,
        )
        self.filename = filename

    @cached_property
    def model_obj(self) -> ModelObj:
        """Allows for lazy loading of underlying model weights."""
        return self._load_model()

    def _load_model(self) -> ModelObj:
        from llama_cpp import Llama

        if "chat_template" not in self.config:
            self.config["chat_template"] = infer_chat_template(self.model_name_or_path)

        if self.model_name_or_path:
            llama = Llama.from_pretrained(
                repo_id=self.model_name_or_path,
                filename=self.filename,
                verbose=False,
                **self.config,
            )
        else:
            llama = Llama(self.filename, verbose=False, **self.config)

        lm = LlamaCpp(
            llama,
            echo=False,
            device_map=self.config.pop("device_map", None),
            **self.config,
        )
        # Try to infer if we're in chat mode
        if lm.engine.tokenizer._orig_tokenizer.chat_template is None:
            logger.debug(
                Fore.YELLOW
                + "chat_template not found in tokenizer config.\nBlendSQL currently only works with chat models"
                + Fore.RESET
            )
        self.tokenizer = llama.tokenizer()
        return lm
