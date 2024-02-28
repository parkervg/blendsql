"""
Contains base class for guidance programs for LLMs.
https://github.com/guidance-ai/guidance
"""
from typing import Optional
from guidance.models import Model, Chat
from guidance import user, system, assistant
from contextlib import nullcontext
import inspect


class Program:
    def __new__(
        self,
        model: Model,
        gen_kwargs: Optional[dict] = None,
        few_shot: bool = True,
        **kwargs,
    ):
        self.model = model
        self.gen_kwargs = gen_kwargs if gen_kwargs is not None else {}
        self.few_shot = few_shot
        assert isinstance(
            self.model, Model
        ), f"GuidanceProgram needs a guidance.models.Model object!\nGot {type(self.model)}"
        (
            self.usercontext,
            self.systemcontext,
            self.assistantcontext,
        ) = self._get_contexts(self.model)
        return self.__call__(self, **kwargs)

    def __call__(self, *args, **kwargs):
        pass

    def __str__(self):
        return inspect.getsource(self.__call__)

    @staticmethod
    def _get_contexts(model: Model):
        usercontext = nullcontext()
        systemcontext = nullcontext()
        assistantcontext = nullcontext()
        if isinstance(model, Chat):
            usercontext = user()
            systemcontext = system()
            assistantcontext = assistant()
        return (usercontext, systemcontext, assistantcontext)
