import functools
from typing import Callable, List
import guidance
from attr import attrib, attrs
from pathlib import Path
from dotenv import load_dotenv
from colorama import Fore
import time
import threading
import re
from diskcache import Cache
import platformdirs
import hashlib

from blendsql._programs import GuidanceProgram

class TokenTimer(threading.Thread):
    """Class to handle refreshing tokens."""

    def __init__(self, init_fn, refresh_interval_min: int = 30):
        super().__init__()
        self.daemon = True
        self.refresh_interval_min = refresh_interval_min
        self.init_fn = init_fn

    def run(self):
        self.init_fn()
        while True:
            time.sleep(self.refresh_interval_min * 60)
            print(Fore.YELLOW + "Refreshing the access tokens..." + Fore.RESET)
            self.init_fn()


@attrs
class LLM:
    """Parent class for all LLM endpoints"""

    model_name_or_path: str = attrib()
    tokenizer: Callable = attrib(default=None)
    requires_config: bool = attrib(default=False)
    refresh_interval_min: int = attrib(default=None)
    env: str = attrib(default=".")

    model: guidance.models.Model = attrib(init=False)
    prompts: list = attrib(init=False)
    cache: Cache = attrib(init=False)

    gen_kwargs: dict = {}
    num_llm_calls: int = 0
    num_prompt_tokens: int = 0

    def __attrs_post_init__(self):
        self.cache = Cache(
            Path(platformdirs.user_cache_dir("blendsql")) / f"{self.__class__}_{self.model_name_or_path}.diskcache"
        )
        self.prompts: List[str] = []
        if self.requires_config:
            if self.env is None:
                self.env = "."
            _env = Path(self.env)
            env_filepath = _env / ".env" if _env.is_dir() else _env
            if env_filepath.is_file():
                load_dotenv()
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
        self._setup()
        self.model = self._load_model()

    def predict(self, program: GuidanceProgram, **kwargs) -> dict:
        key = self._create_key(program, **kwargs)
        if key in self.cache:
            return self.cache.get(key)
        # Modify fields used for tracking LLM usage
        self.num_llm_calls += 1
        model = program(model=self.model, **kwargs)
        if self.tokenizer is not None:
            prompt = re.sub(
                r"(?<=\>)(assistant|user|system)", "", model._current_prompt()
            )
            prompt = re.sub(r"\<.*?\>", "", prompt)
            self.num_prompt_tokens += len(self.tokenizer.encode(prompt))
            self.prompts.append(prompt)
        self.cache[key] = model._variables
        return model._variables

    def _create_key(self, program: GuidanceProgram, **kwargs):
        hasher = hashlib.md5()
        # Ignore partials, which create a random key within session
        options_str = str(sorted([(k, v) for k, v in kwargs.items() if not isinstance(v, functools.partial)]))
        combined = "{}{}".format(str(program), options_str).encode()
        hasher.update(combined)
        return hasher.hexdigest()

    def _setup(self, **kwargs) -> None:
        ...

    def _load_model(self) -> guidance.models.Model:
        ...
