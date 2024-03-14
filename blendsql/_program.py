"""
Contains base class for guidance programs for LLMs.
https://github.com/guidance-ai/guidance
"""
from typing import Optional
from guidance.models import Model, Chat
from guidance import user, system, assistant
from contextlib import nullcontext
import inspect
import ast
import textwrap


def get_contexts(model: Model):
    usercontext = nullcontext()
    systemcontext = nullcontext()
    assistantcontext = nullcontext()
    if isinstance(model, Chat):
        usercontext = user()
        systemcontext = system()
        assistantcontext = assistant()
    return (usercontext, systemcontext, assistantcontext)


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
        ) = get_contexts(self.model)
        return self.__call__(self, **kwargs)

    def __call__(self, *args, **kwargs):
        pass


def program_to_str(program: Program):
    """Create a string representation of a program.
    It is slightly tricky, since in addition to getting the code content, we need to
        1) identify all global variables referenced within a function, and then
        2) evaluate the variable value
    This is required, since if we have some global constant `PROMPT` called,
    we don't want to fetch from a previously created cache if the value of `PROMPT` changes.

    To avoid extreme messiness, we don't traverse into globals pointing at functions.

    Example:
        >>> PROMPT = "Here is my question: {question}"
        >>> class CorrectionProgram(Program):
        >>>     def __call__(self, question: str, **kwargs):
        >>>         return self.model + PROMPT.format(question)

    Some helpful refs:
        - https://github.com/universe-proton/universe-topology/issues/15
    """
    call_content = textwrap.dedent(inspect.getsource(program.__call__))
    root = ast.parse(call_content)
    root_names = {node.id for node in ast.walk(root) if isinstance(node, ast.Name)}
    co_varnames = set(program.__call__.__code__.co_varnames)
    names_to_resolve = sorted(root_names.difference(co_varnames))
    resolved_names = ""
    if len(names_to_resolve) > 0:
        globals_as_dict = dict(inspect.getmembers(program.__call__))["__globals__"]
        for name in names_to_resolve:
            if name in globals_as_dict:
                val = globals_as_dict[name]
                # Ignore functions
                if not callable(val):
                    resolved_names += f"{val}\n"
    return f"{call_content}{resolved_names}"
