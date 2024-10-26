"""Abstract class, for use with all ingredients which implement few-shot learning via the `Model` class."""
from abc import abstractmethod


class Example:
    @abstractmethod
    def to_string(self, *args, **kwargs) -> str:
        ...

    # def __str__(self):
    #     """This is needed to ensure we handle caching correctly in _model.py"""
    #     s = ""
    #     for a in dir(self):
    #         if a.startswith("__"):
    #             continue
    #         v = getattr(self, a)
    #         if isinstance(v, set):
    #             s += f"{a}={str(sorted(v))}"
    #         else:
    #             s += f"{a}={v}"
    #     return s
