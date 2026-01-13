import os
import importlib.util

from blendsql.common.logger import logger, Color
from blendsql.models.model import ConstrainedModel, ModelObj

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_KWARGS = {"do_sample": False}

_has_transformers = importlib.util.find_spec("transformers") is not None
_has_torch = importlib.util.find_spec("torch") is not None


class TransformersLLM(ConstrainedModel):
    """Class for Transformers local Model.

    Args:
        model_name_or_path: Name of the model on HuggingFace, or the path to a local model
        caching: Bool determining whether we access the model's cache
        config: Additional parameters to pass to the `from_pretrained()` call

    Examples:
        ```python
        from blendsql.models import TransformersLLM

        model = TransformersLLM(
            "Qwen/Qwen1.5-0.5B",
            config={"device_map": "auto", "torch_dtype": torch.bfloat16},
        )
        ```
    """

    def __init__(
        self,
        model_name_or_path: str,
        config: dict | None = None,
        caching: bool = False,
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

    def _load_model(self) -> ModelObj:
        # https://huggingface.co/blog/how-to-generate
        # from guidance.models._transformers import Transformers
        from guidance.models import Transformers

        lm = Transformers(
            self.model_name_or_path,
            echo=False,
            device_map=self.config.pop("device_map", None),
            **self.config,
        )
        # Try to infer if we're in chat mode
        if lm.engine.tokenizer._orig_tokenizer.chat_template is None:
            logger.debug(
                Color.error(
                    "chat_template not found in tokenizer config.\nBlendSQL currently only works with chat models"
                )
            )
        return lm

    def tokenizer_encode(self, s: str):
        """Override the tokenizer.encode call to NOT add special tokens"""
        return self.tokenizer.encode(s, add_special_tokens=False)


class LlamaCpp(ConstrainedModel):
    """Class for LlamaCpp local Model.

    Args:
        filename: Specific .gguf file (local or on HuggingFace) to load
        model_name_or_path: Optional path to the model on HuggingFace
        caching: Bool determining whether we access the model's cache
        config: Additional parameters to pass to the `Llama()` construction call

    Examples:
        ```python
        from blendsql.models import LlamaCpp

        model = LlamaCpp(
            filename="google_gemma-3-12b-it-Q6_K.gguf",
            model_name_or_path="bartowski/google_gemma-3-12b-it-GGUF",
            config={"n_gpu_layers": -1, "n_ctx": 8000, "seed": 100, "n_threads": 16},
        )
        ```
    """

    def __init__(
        self,
        filename: str,
        model_name_or_path: str | None = None,
        config: dict | None = None,
        caching: bool = False,
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
        model_name_or_path: str | None,
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

    def _load_model(self) -> ModelObj:
        from guidance.models import LlamaCpp as GuidanceLlamaCpp
        import logging

        logging.getLogger("guidance").setLevel(logging.CRITICAL)
        logging.getLogger("llama_cpp").setLevel(logging.CRITICAL)

        # llama.cpp doesn't like when we have two running simultaneously
        #   so we do a little switcheroo with the tokenizer here
        if hasattr(self, "tokenizer"):
            self.__delattr__("tokenizer")

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


class ConstrainedLiteLLM(ConstrainedModel):
    def __init__(
        self,
        model_name_or_path: str,
        config: dict | None = None,
        caching: bool = False,
        **kwargs,
    ):
        print("This guidance model is experimental! Use with caution.")
        if config is None:
            config = {}

        super().__init__(
            model_name_or_path=model_name_or_path,
            requires_config=False,
            config=config,
            caching=caching,
            **kwargs,
        )

        class DummyTokenizer:
            def encode(self, *args, **kwargs):
                return ["test"]

        self.tokenizer = DummyTokenizer()

    def _load_model(self, *args, **kwargs) -> ModelObj:
        import guidance

        lm = guidance.models.experimental.LiteLLM(
            {
                "model_name": self.model_name_or_path,
                "litellm_params": self.config,
            },
            echo=False,
        )
        return lm
