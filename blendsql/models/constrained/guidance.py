import os
import importlib.util
from typing import Optional
from colorama import Fore
from functools import cached_property

from blendsql.common.logger import logger
from blendsql.models.model import ConstrainedModel, ModelObj

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

    elif "qwen2.5" in model_name_or_path.lower():
        # https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/tokenizer_config.json
        # Uses ChatML
        from guidance.chat import ChatMLTemplate

        chat_template = ChatMLTemplate

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
        # from guidance.models._transformers import Transformers
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
        filename: str,
        model_name_or_path: Optional[str] = None,
        config: Optional[dict] = None,
        caching: bool = True,
        **kwargs,
    ):
        if config is None:
            config = {}

        super().__init__(
            model_name_or_path=model_name_or_path,  # type: ignore
            requires_config=False,
            tokenizer=self._load_llama_cpp(
                filename=filename,
                model_name_or_path=model_name_or_path,
                config=config,
                vocab_only=True,
            ).tokenizer_,
            config=config,
            caching=caching,
            **kwargs,
        )
        self.filename = filename

    @staticmethod
    def _load_llama_cpp(
        filename: str,
        model_name_or_path: Optional[str],
        config: dict,
        vocab_only: bool = False,
    ):
        from llama_cpp import Llama

        if model_name_or_path:
            _config = config if not vocab_only else {}
            model = Llama.from_pretrained(
                repo_id=model_name_or_path,
                filename=filename,
                verbose=False,
                vocab_only=vocab_only,
                **_config,
            )
        else:
            model = Llama(filename, verbose=False, vocab_only=vocab_only, **config)

        # https://github.com/abetlen/llama-cpp-python/issues/1610
        import atexit

        @atexit.register
        def free_model():
            model.close()

        return model

    @cached_property
    def model_obj(self) -> ModelObj:
        """Allows for lazy loading of underlying model weights."""
        return self._load_model()

    def _load_model(self) -> ModelObj:
        from guidance.models import LlamaCpp as GuidanceLlamaCpp
        import logging

        logging.getLogger("guidance").setLevel(logging.CRITICAL)
        logging.getLogger("llama_cpp").setLevel(logging.CRITICAL)

        # llama.cpp doesn't like when we have two running simultaneously
        #   so we do a little switcheroo with the tokenizer here
        self.__delattr__("tokenizer")

        if "chat_template" not in self.config:
            self.config["chat_template"] = infer_chat_template(self.model_name_or_path)

        lm = GuidanceLlamaCpp(
            self._load_llama_cpp(
                filename=self.filename,
                model_name_or_path=self.model_name_or_path,
                config=self.config,
            ),
            echo=False,
            chat_template=self.config.get("chat_template"),
        )
        self.tokenizer = lm.engine.model_obj.tokenizer_
        return lm
