import functools
from typing import Any, List
import guidance
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

from .._program import Program, program_to_str
from .._constants import IngredientKwarg
from ..db.utils import truncate_df_content

CONTEXT_TRUNCATION_LIMIT = 100


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
    refresh_interval_min: int = attrib(default=None)
    env: str = attrib(default=".")
    caching: bool = attrib(default=True)

    model: guidance.models.Model = attrib(init=False)
    prompts: list = attrib(init=False)
    cache: Cache = attrib(init=False)
    run_setup_on_load: bool = attrib(default=True)

    gen_kwargs: dict = {}
    num_llm_calls: int = 0
    num_prompt_tokens: int = 0

    def __attrs_post_init__(self):
        if self.caching:
            self.cache = Cache(
                Path(platformdirs.user_cache_dir("blendsql"))
                / f"{self.model_name_or_path}.diskcache"
            )
        self.prompts: List[str] = []
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
        self.model = self._load_model()

    def predict(self, program: Program, **kwargs) -> dict:
        """Takes a `Program` and some kwargs, and evaluates it with context of
        current Model.

        Args:
            program: guidance program used to generate Model output
            **kwargs: any additional kwargs will get passed to the program

        Returns:
            dict containing all Model variable names and their values.

        Examples:
            >>> llm.predict(program, **kwargs)
            {"result": '"This is Model generated output"'}
        """
        if self.caching:
            # First, check our cache
            key = self._create_key(program, **kwargs)
            if key in self.cache:
                self.prompts.insert(
                    -1, self.format_prompt(self.cache.get(key), **kwargs)
                )
                return self.cache.get(key)
        # Modify fields used for tracking Model usage
        self.num_llm_calls += 1
        model = program(model=self.model, **kwargs)
        if self.tokenizer is not None:
            prompt = model._current_prompt()
            self.num_prompt_tokens += len(self.tokenizer.encode(prompt))
        self.prompts.insert(-1, self.format_prompt(model._variables, **kwargs))
        if self.caching:
            self.cache[key] = model._variables
        return model._variables

    def _create_key(self, program: Program, **kwargs) -> str:
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
                    if not isinstance(v, functools.partial)
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

    @staticmethod
    def format_prompt(res, **kwargs) -> dict:
        d = {"answer": res}
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
    def _setup(self, **kwargs) -> None:
        """Any additional setup required to get this Model up and functioning
        should go here. For example, in the AzureOpenaiLLM, we have some logic
        to refresh our client secrets every 30 min.
        """
        ...

    @abstractmethod
    def _load_model(self) -> Any:
        """Logic for instantiating the guidance model class goes here."""
        ...
