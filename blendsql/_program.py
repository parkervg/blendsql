"""
Contains base class for guidance programs for LLMs.
https://github.com/guidance-ai/guidance
"""
from typing import Optional
from guidance.models import Model as GuidanceModel
from guidance.models import Chat as GuidanceChatModel
from guidance import user, system, assistant
from contextlib import nullcontext
import inspect
import ast
import textwrap


def get_contexts(model: GuidanceModel):
    usercontext = nullcontext()
    systemcontext = nullcontext()
    assistantcontext = nullcontext()
    if isinstance(model, GuidanceChatModel):
        usercontext = user()
        systemcontext = system()
        assistantcontext = assistant()
    return (usercontext, systemcontext, assistantcontext)


class Program:
    """
    TODO: how to add streaming?
        if isinstance(res, ModelStream):
        # Fetch actual model by iterating
        # https://github.com/guidance-ai/guidance/blob/main/tests/models/test_model.py#L85
        for idx, event in enumerate(res):
            print(Fore.LIGHTCYAN_EX + str(event) + Fore.RESET)
            res = event
    """

    def __new__(
        self,
        model: GuidanceModel,
        gen_kwargs: Optional[dict] = None,
        few_shot: bool = True,
        **kwargs,
    ):
        self.model = model
        self.gen_kwargs = gen_kwargs
        self.gen_kwargs = {} if gen_kwargs is None else gen_kwargs
        self.few_shot = few_shot
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
    source_func = program.__call__
    call_content = textwrap.dedent(inspect.getsource(source_func))
    root = ast.parse(call_content)
    root_names = {node.id for node in ast.walk(root) if isinstance(node, ast.Name)}
    co_varnames = set(source_func.__code__.co_varnames)
    names_to_resolve = sorted(root_names.difference(co_varnames))
    resolved_names = ""
    if len(names_to_resolve) > 0:
        globals_as_dict = dict(inspect.getmembers(source_func))["__globals__"]
        for name in names_to_resolve:
            if name in globals_as_dict:
                val = globals_as_dict[name]
                # Ignore functions - we really only want scalars here
                if not callable(val):
                    resolved_names += f"{val}\n"
    return f"{call_content}{resolved_names}"
