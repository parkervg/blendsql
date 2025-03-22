from typing import Union

from .._model import ModelObj, Model


class LMString(str):
    """Allows us to lazily load a guidance model_obj.
    Until we have reason to generate, we instead append to this
        str subclass, and utilize our cache if possible.
    """

    def __new__(cls, content="", variables=None):
        instance = super().__new__(cls, content)
        instance._variables = variables if variables is not None else {}
        return instance

    def _current_prompt(self):
        return str(self)

    def __add__(self, other):
        new_content = super().__add__(other)
        return LMString(new_content, self._variables)

    def __getitem__(self, item):
        return self._variables[item]


def maybe_load_lm(model: Model, lm: Union[LMString, ModelObj]) -> ModelObj:
    if isinstance(lm, LMString):
        new_lm = model.model_obj + lm
        new_lm._variables = lm._variables
        return new_lm
    return lm
