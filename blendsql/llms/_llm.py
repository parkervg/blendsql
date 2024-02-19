from abc import ABC, abstractmethod
from typing import Union, Callable
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

    modelclass: guidance.models._model = attrib()
    model_name_or_path: str = attrib()
    tokenizer: Union[tiktoken.Encoding, "transformers.Tokenizer"] = attrib(default=None)
    requires_config: bool = attrib(default=False)
    refresh_interval_min: int = attrib(default=None)
    model: guidance.models.Model = attrib(init=False)

    gen_kwargs: dict = {}
    num_llm_calls: int = 0
    num_prompt_tokens: int = 0

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
        # Instantiate model class after maybe loading creds
        self.model = self.modelclass(self.model_name_or_path, echo=False)

    def predict(self, program: Callable, **kwargs) -> dict:
        self.num_llm_calls += 1
        model = program(model=self.model, **kwargs)
        if self.tokenizer is not None:
            prompt = re.sub(
                r"(?<=\>)(assistant|user|system)", "", model._current_prompt()
            )
            prompt = re.sub(r"\<.*?\>", "", prompt)
            self.num_prompt_tokens += len(self.tokenizer.encode(prompt))
        return model._variables

    @abstractmethod
    def _setup(self, **kwargs) -> None:
        ...
