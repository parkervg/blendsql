from abc import ABC, abstractmethod
from typing import Dict, Union
import guidance
import tiktoken
from attr import attrib, attrs
import os
from dotenv import load_dotenv
from colorama import Fore
import time
import threading
import re


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
class LLM(ABC):
    """Abstract class for all LLM endpoints"""

    model_name_or_path: str = attrib()
    requires_config: bool = attrib(default=False)
    refresh_interval_min: int = attrib(default=None)
    encoding: tiktoken.Encoding = attrib(default=None)

    gen_kwargs: dict = {}
    num_llm_calls: int = 0
    num_tokens_passed: int = 0

    def __attrs_post_init__(self):
        if self.requires_config:
            if os.path.exists(".env"):
                load_dotenv()
            else:
                raise FileNotFoundError(
                    f"{self.__class__} requires a .env file to be present in the directory with necessary environment variables"
                )
        self._setup()
        if self.refresh_interval_min:
            timer = TokenTimer(init_fn=self._setup)
            timer.start()

    def predict(self, program: str, **kwargs) -> Union[str, Dict]:
        self.num_llm_calls += 1
        if self.encoding is not None:
            self.num_tokens_passed += len(
                self.encoding.encode(re.sub(r"\{\{.*?\}\}", "", program.__str__()))
            )
            for _k, v in kwargs.items():
                if v is None:
                    continue
                if not isinstance(v, bool):
                    self.num_tokens_passed += len(self.encoding.encode(str(v)))
        program: guidance._program.Program = guidance(program, silent=True)
        return self._predict(program, **kwargs)

    @abstractmethod
    def _predict(
        self, program: guidance._program.Program, **kwargs
    ) -> Union[str, Dict]:
        ...

    @abstractmethod
    def _setup(self, **kwargs) -> None:
        ...
