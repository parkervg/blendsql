from typing import Union
import copy
from dataclasses import dataclass

from ..model import ModelObj, Model


class LMString(str):
    """Allows us to lazily load a guidance model_obj.
    Until we have reason to generate, we instead append to this
        str subclass, and utilize our cache if possible.
    """

    def __new__(cls, content="", variables=None):
        instance = super().__new__(cls, content)
        instance._variables = variables if variables is not None else {}
        return instance

    def set(self, key, value):
        new_lm = copy.copy(self)
        new_lm._variables[key] = value
        return new_lm

    def __add__(self, other):
        new_content = super().__add__(other)
        return LMString(new_content, self._variables)

    def __getitem__(self, item):
        return self._variables[item]

    def _get_usage(self):
        @dataclass
        class DummyOutput:
            input_tokens: int

        return DummyOutput(len(str(self)))


def maybe_load_lm(model: Model, lm: Union[LMString, ModelObj]) -> ModelObj:
    if isinstance(lm, LMString):
        new_lm = model.model_obj + lm
        for k, v in lm._variables.items():
            new_lm = new_lm.set(k, v)
        return new_lm
    return lm
