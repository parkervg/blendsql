from abc import ABC, abstractmethod
import os
from typing import Dict, Union
import guidance
from attr import attrib, attrs
from blendsql.ingredients.builtin.llm.llm import LLM
from dotenv import load_dotenv
from colorama import Fore
import time
import threading


class TokenTimer(threading.Thread):
    """Class to handle refreshing tokens."""

    def __init__(self, init_fn, requires_config: bool = True, **kwargs):
        super().__init__()
        self.daemon = True
        self.refresh_interval_min = kwargs['refresh_interval_min']
        self.init_fn = init_fn
        self.requires_config = requires_config
        self.kwargs = kwargs

    def run(self):
        while True:
            print(Fore.YELLOW + "Refreshing the access tokens..." + Fore.RESET)
            self.init_fn(self.requires_config, **self.kwargs)
            time.sleep(self.refresh_interval_min * self.refresh_interval_min)

@attrs
class RemoteLLM(LLM):
    """Abstract class for all Remote LLMs"""

    is_connected: bool = False

    def setup(self, requires_config: bool = True, **kwargs) -> None:
        if requires_config:
            if os.path.exists('.env'):
                load_dotenv()
            else:
                raise FileNotFoundError(f"{self.__class__} requires a .env file to be present in the directory with necessary environment variables")
        self._setup(requires_config, **kwargs)
        if "refresh_interval_min" in kwargs and kwargs["refresh_interval_min"] > 0:
            timer = TokenTimer(self._setup, requires_config, **kwargs)
            timer.start()

    @abstractmethod
    def _setup(self, requires_config: bool = True, **kwargs) -> None:
        ...