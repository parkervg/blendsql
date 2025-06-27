from typing import (
    Any,
    List,
    Optional,
    Generic,
    Dict,
    TypeVar,
    Sequence,
    Callable,
    Tuple,
)
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from colorama import Fore
import time
import threading
from diskcache import Cache
import platformdirs
import hashlib
import inspect
from textwrap import dedent
from abc import abstractmethod
from attr import attrs, attrib

from ..db.utils import truncate_df_content
from blendsql.common.logger import logger

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
            print(Fore.YELLOW + "Refreshing the access tokens..." + Fore.RESET)
            self.init_fn()


@attrs
class Model:
    """Parent class for all BlendSQL Models."""

    model_name_or_path: str = attrib()
    tokenizer: Any = attrib(default=None)
    requires_config: bool = attrib(default=False)
    refresh_interval_min: Optional[int] = attrib(default=None)
    config: dict = attrib(default=None)
    env: str = attrib(default=".")
    caching: bool = attrib(default=True)

    model_obj: Generic[ModelObj] = attrib(init=False)
    prompts: List[dict] = attrib(factory=list)
    raw_prompts: List[str] = attrib(factory=list)
    cache: Optional[Cache] = attrib(default=None)
    run_setup_on_load: bool = attrib(default=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    num_generation_calls: int = 0

    def __attrs_post_init__(self):
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
        self, *args, funcs: Optional[Sequence[Callable]] = None, **kwargs
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
        self, *args, funcs: Optional[Sequence[Callable]] = None, **kwargs
    ) -> Tuple[Any, str]:
        response: Dict[str, str] = None  # type: ignore
        key: str = self._create_key(funcs=funcs, *args, **kwargs)
        if key in self.cache:
            logger.debug(Fore.MAGENTA + "Using model cache..." + Fore.RESET)
            response = self.cache.get(key)  # type: ignore
        return (response, key)

    def reset_stats(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.num_generation_calls = 0

    @staticmethod
    def format_prompt(response: str, **kwargs) -> dict:
        d: Dict[str, Any] = {"answer": response}
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
    pass


class UnconstrainedModel(Model):
    pass
