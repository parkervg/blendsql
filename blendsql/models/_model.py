from typing import Any, List, Optional, Generic, Type, Dict, TypeVar, Union
import pandas as pd
from attr import attrib, attrs
from pathlib import Path
from dotenv import load_dotenv
from colorama import Fore
import time
import threading
from diskcache import Cache
import platformdirs
import hashlib
from abc import abstractmethod
from functools import cached_property

from .._logger import logger
from .._program import Program, program_to_str
from .._constants import IngredientKwarg
from ..db.utils import truncate_df_content

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
    load_model_kwargs: dict = attrib(default=None)
    env: str = attrib(default=".")
    caching: bool = attrib(default=True)

    model_obj: Generic[ModelObj] = attrib(init=False)
    prompts: List[dict] = attrib(init=False)
    prompt_tokens: int = attrib(init=False)
    completion_tokens: int = attrib(init=False)
    num_calls: int = attrib(init=False)
    cache: Cache = attrib(init=False)
    run_setup_on_load: bool = attrib(default=True)

    def __attrs_post_init__(self):
        if self.caching:
            self.cache = Cache(
                Path(platformdirs.user_cache_dir("blendsql"))
                / f"{self.model_name_or_path}.diskcache"
            )
        if self.load_model_kwargs is None:
            self.load_model_kwargs = {}
        self.prompts: List[dict] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.num_calls = 0
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

    def predict(self, program: Type[Program], **kwargs) -> str:
        """Takes a `Program` and some kwargs, and evaluates it with context of
        current Model.

        Args:
            program: The `Program` object used to generate Model output
            **kwargs: any additional kwargs will get passed to the program

        Returns:
            dict containing all Model variable names and their values.

        Examples:
            >>> model.predict(program, **kwargs)
            "This is model generated output"
        """
        if self.caching:
            # First, check our cache
            key: str = self._create_key(program, **kwargs)
            if key in self.cache:
                logger.debug(Fore.MAGENTA + "Using cache..." + Fore.RESET)
                response: str = self.cache.get(key)  # type: ignore
                self.prompts.insert(-1, self.format_prompt(response, **kwargs))
                return response
        # Modify fields used for tracking Model usage
        response: Union[str, List[str]]
        prompt: str
        response, prompt = program(model=self, **kwargs)
        self.prompts.insert(-1, self.format_prompt(response, **kwargs))
        self.num_calls += 1
        if self.tokenizer is not None:
            self.prompt_tokens += len(self.tokenizer.encode(prompt))
            self.completion_tokens += sum(
                [len(self.tokenizer.encode(r)) for r in " ".join(response)]
            )
        if self.caching:
            self.cache[key] = response  # type: ignore
        return response

    def _create_key(self, program: Type[Program], **kwargs) -> str:
        """Generates a hash to use in diskcache Cache.
        This way, we don't need to send our prompts to the same Model
        if our context of Model + program + kwargs is the same.

        Returns:
            md5 hash used as key in diskcache
        """
        hasher = hashlib.md5()
        # Ignore partials, which create a random key within session
        options_str = str(
            sorted(
                [
                    (k, sorted(v) if isinstance(v, set) else v)
                    for k, v in kwargs.items()
                    if not callable(v)
                ]
            )
        )
        combined = "{}||{}||{}".format(
            f"{self.model_name_or_path}||{type(self)}",
            program_to_str(program),
            options_str,
        ).encode()
        hasher.update(combined)
        return hasher.hexdigest()

    @cached_property
    def model_obj(self) -> ModelObj:
        """Allows for lazy loading of underlying model weights."""
        return self._load_model()

    @staticmethod
    def format_prompt(response: str, **kwargs) -> dict:
        d: Dict[str, Any] = {"answer": response}
        if IngredientKwarg.QUESTION in kwargs:
            d[IngredientKwarg.QUESTION] = kwargs.get(IngredientKwarg.QUESTION)
        if IngredientKwarg.CONTEXT in kwargs:
            context = kwargs.get(IngredientKwarg.CONTEXT)
            if isinstance(context, pd.DataFrame):
                context = truncate_df_content(context, CONTEXT_TRUNCATION_LIMIT)
                d[IngredientKwarg.CONTEXT] = context.to_dict(orient="records")
        if IngredientKwarg.VALUES in kwargs:
            d[IngredientKwarg.VALUES] = kwargs.get(IngredientKwarg.VALUES)
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
        Will most likely be an outlines model object,
        but in some cases (like OllamaLLM) we make an exception.
        """
        ...


class LocalModel(Model):
    pass


class RemoteModel(Model):
    pass
