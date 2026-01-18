from typing import Any, Generic, TypeVar, Sequence, Callable, Tuple, Union

import guidance
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import time
import threading
from diskcache import Cache
import platformdirs
import hashlib
import inspect
from textwrap import dedent
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property

from ..db.utils import truncate_df_content
from blendsql.common.logger import logger, Color

CONTEXT_TRUNCATION_LIMIT = 100
ModelObj = TypeVar("ModelObj")


class TokenTimer(threading.Thread):
    """Class to handle refreshing tokens."""

    def __init__(self, init_fn, refresh_interval_min: int = 30):
        super().__init__()
        self.daemon = True
        self.refresh_interval_min = refresh_interval_min
        self.init_fn = init_fn

    def run(self):
        while True:
            time.sleep(self.refresh_interval_min * 60)
            print(Color.warning("Refreshing the access tokens..."))
            self.init_fn()


@dataclass
class Model:
    """Parent class for all BlendSQL Models."""

    model_name_or_path: str = field()
    tokenizer: Any = field(default=None)
    requires_config: bool = field(default=False)
    refresh_interval_min: int | None = field(default=None)
    config: dict = field(default=None)
    env: str = field(default=".")
    caching: bool = field(default=False)

    model_obj: Generic[ModelObj] = field(init=False)
    maybe_add_system_prompt: Callable = field(
        default=lambda lm: lm
    )  # For ConstrainedModels only
    prompts: list[dict] = field(default_factory=list)
    raw_prompts: list[str] = field(default_factory=list)
    cache: Cache | None = field(default=None)
    run_setup_on_load: bool = field(default=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    num_generation_calls: int = 0
    num_cache_hits: int = 0

    def __post_init__(self):
        self.cache = Cache(
            Path(platformdirs.user_cache_dir("blendsql"))
            / f"{self.model_name_or_path}.diskcache"
        )
        if self.config is None:
            self.config = {}
        if self.requires_config:
            if self.env is None:
                self.env = "."
            _env = Path(self.env)
            env_filepath = _env / ".env" if _env.is_dir() else _env
            if env_filepath.is_file():
                load_dotenv(str(env_filepath))
            else:
                raise FileNotFoundError(
                    f"{self.__class__} requires a .env file to be present at '{env_filepath}' with necessary environment variables\nPut it somewhere else? Use the `env` argument to point me to the right directory."
                )
        if self.refresh_interval_min:
            timer = TokenTimer(
                init_fn=self._setup, refresh_interval_min=self.refresh_interval_min
            )
            timer.start()
        if self.tokenizer is not None:
            assert hasattr(self.tokenizer, "encode") and callable(
                self.tokenizer.encode
            ), f"`tokenizer` passed to {self.__class__} should have `encode` method!"
        if self.run_setup_on_load:
            self._setup()

    def _create_key(
        self, *args, funcs: Sequence[Callable] | None = None, **kwargs
    ) -> str:
        """Generates a hash to use in diskcache Cache.
        This way, we don't need to send our prompts to the same Model
        if our context of Model + args + kwargs is the same.

        Returns:
            md5 hash used as key in diskcache
        """
        hasher = hashlib.md5()
        params_str = ""
        if len(kwargs) > 0:
            params_str += str(sorted([(k, str(v)) for k, v in kwargs.items()]))
        if len(args) > 0:
            params_str += str([arg for arg in args])
        if funcs:
            params_str += "\n".join([dedent(inspect.getsource(func)) for func in funcs])
        combined_str = "{}||{}".format(
            f"{self.model_name_or_path}||{type(self)}",
            params_str,
        ).encode()
        hasher.update(combined_str)
        return hasher.hexdigest()

    def check_cache(
        self, *args, funcs: Sequence[Callable] | None = None, **kwargs
    ) -> Tuple[Any, str]:
        response: dict[str, str] = None  # type: ignore
        key: str = self._create_key(funcs=funcs, *args, **kwargs)
        if key in self.cache:
            self.num_cache_hits += 1
            logger.debug(
                Color.model_or_data_update(
                    f"Using model cache ({self.num_cache_hits})..."
                )
            )
            response = self.cache.get(key)  # type: ignore
        return (response, key)

    def reset_stats(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.num_generation_calls = 0
        self.num_cache_hits = 0

    def tokenizer_encode(self, s: str):
        return self.tokenizer.encode(s)

    @staticmethod
    def format_prompt(response: str, **kwargs) -> dict:
        d: dict[str, Any] = {"answer": response}
        if "question" in kwargs:
            d["question"] = kwargs.get("question")
        if "context" in kwargs:
            context = kwargs.get("context")
            if isinstance(context, pd.DataFrame):
                context = truncate_df_content(context, CONTEXT_TRUNCATION_LIMIT)
                d["context"] = context.to_dict(orient="records")
        if "values" in kwargs:
            d["values"] = kwargs.get("values")
        return d

    @abstractmethod
    def _setup(self, *args, **kwargs) -> None:
        """Any additional setup required to get this Model up and functioning
        should go here. For example, in the AzureOpenaiLLM, we have some logic
        to refresh our client secrets every 30 min.
        """
        ...

    @abstractmethod
    def _load_model(self, *args, **kwargs) -> ModelObj:
        """Logic for instantiating the model class goes here.
        Will most likely be a guidance model object,
        but in some cases (like OllamaLLM) we make an exception.
        """
        ...


class ConstrainedModel(Model):
    @staticmethod
    def infer_chat_template(
        model_name_or_path: str,
    ) -> tuple[Union["ChatTemplate", None], Callable]:
        # Try to infer chat template
        chat_template = None
        maybe_add_system_prompt = lambda lm: lm
        _model_name_or_path = model_name_or_path.split("/")[-1].lower()

        if "smollm" in _model_name_or_path.lower():
            from guidance.chat import ChatMLTemplate

            chat_template = ChatMLTemplate

        elif "llama-3" in _model_name_or_path:
            from guidance.chat import Llama3ChatTemplate

            chat_template = Llama3ChatTemplate

        elif "llama-2" in _model_name_or_path:
            from guidance.chat import Llama2ChatTemplate

            chat_template = Llama2ChatTemplate

        elif "phi-3" in _model_name_or_path:
            from guidance.chat import Phi3MiniChatTemplate

            chat_template = Phi3MiniChatTemplate

        elif "qwen2.5" in _model_name_or_path:
            from guidance.chat import Qwen2dot5ChatTemplate

            chat_template = Qwen2dot5ChatTemplate

            def maybe_add_system_prompt(lm):
                with guidance.system():
                    lm += "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                return lm

        elif "qwen3" in _model_name_or_path:
            from guidance.chat import Qwen3ChatTemplate

            chat_template = Qwen3ChatTemplate

        elif "gemma" in _model_name_or_path:
            from guidance.chat import Gemma29BInstructChatTemplate

            chat_template = Gemma29BInstructChatTemplate

        if chat_template is not None:
            logger.debug(
                Color.model_or_data_update(
                    f"Loading '{model_name_or_path}' with '{chat_template.__name__}' chat template..."
                )
            )
        return (chat_template, maybe_add_system_prompt)

    @cached_property
    def model_obj(self) -> ModelObj:
        """Allows for lazy loading of underlying model weights."""
        if "chat_template" not in self.config:
            filename = None
            if hasattr(self, "filename"):
                filename = self.filename
            (
                self.config["chat_template"],
                self.maybe_add_system_prompt,
            ) = self.infer_chat_template(self.model_name_or_path or filename)
        return self._load_model()


class UnconstrainedModel(Model):
    pass
