"""Abstract class, for use with all ingredients which implement few-shot learning via the `Model` class."""
from abc import abstractmethod


class Example:
    @abstractmethod
    def to_string(self, *args, **kwargs) -> str:
        ...
