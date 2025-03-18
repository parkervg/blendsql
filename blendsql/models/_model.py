from typing import Any, List, Optional, Generic, Dict, TypeVar
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

from .._constants import IngredientKwarg
from ..db.utils import truncate_df_content
from ..ingredients.few_shot import Example

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


def serialize(v: Any) -> str:
    if isinstance(v, set):
        return sorted(v)
    elif isinstance(v, Example):
        return v.to_string(context_formatter=lambda df: df.to_markdown(index=False))
    elif isinstance(v, list) and isinstance(v[0], Example):
        return "\n".join(
            [
                _v.to_string(context_formatter=lambda df: df.to_markdown(index=False))
                for _v in v
            ]
        )
    return str(v)


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
    cache: Cache = attrib(init=False)
    run_setup_on_load: bool = attrib(default=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    num_calls: int = 0

    def __attrs_post_init__(self):
        if self.caching:
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

    def _create_key(self, *args, **kwargs) -> str:
        """Generates a hash to use in diskcache Cache.
        This way, we don't need to send our prompts to the same Model
        if our context of Model + kwargs is the same.

        Returns:
            md5 hash used as key in diskcache
        """
        hasher = hashlib.md5()
        params_str = str(sorted([(k, serialize(v)) for k, v in kwargs.items()]))
        combined_str = "{}||{}".format(
            f"{self.model_name_or_path}||{type(self)}",
            params_str,
        ).encode()
        hasher.update(combined_str)
        return hasher.hexdigest()

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
        Will most likely be a guidance model object,
        but in some cases (like OllamaLLM) we make an exception.
        """
        ...

    @abstractmethod
    def _generate(self, *args, **kwargs) -> List[str]:
        ...


class LocalModel(Model):
    pass


class RemoteModel(Model):
    pass
