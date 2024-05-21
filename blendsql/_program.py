"""
Contains base class for guidance programs for LLMs.
https://github.com/guidance-ai/guidance
"""

from typing import Optional, Union
import re
from guidance.models import Model as GuidanceModel
from guidance.models import Chat as GuidanceChatModel
from guidance import user, system, assistant
from guidance import gen
from contextlib import nullcontext
import inspect
import ast
import textwrap
from colorama import Fore

GUIDANCE_TO_OLLAMA_ARGS = {
    "max_tokens": "num_predict",
}


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
    """ """

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
        ...

    @staticmethod
    def gen(
        model: Union[GuidanceModel, "OllamaGuidanceModel"],
        name=None,
        stream: bool = False,
        **kwargs,
    ) -> "GuidanceModel":
        if isinstance(model, GuidanceModel):
            if stream:
                for part in model.stream() + gen(name=name, **kwargs):
                    result = str(part).rsplit("\nBlendSQL:", 1)[-1].strip()
                    result = re.split(re.escape("<|im_start|>assistant\n"), result)[
                        -1
                    ].rstrip("<|im_end|>")
                    print("\n" * 50 + Fore.CYAN + result + Fore.RESET)
                model = model.set("result", result)
            else:
                model += gen(name=name, **kwargs)
        else:
            import ollama
            from ollama import Options

            options = Options(
                {GUIDANCE_TO_OLLAMA_ARGS.get(k, k): v for k, v in kwargs.items()}
            )
            if "temperature" not in options:
                options["temperature"] = 0.0
            response = ollama.chat(
                model=model.model_name_or_path,
                messages=[{"role": "user", "content": model._current_prompt()}],
                options=options,
                stream=stream,
            )
            if stream:
                res = []
                for chunk in response:
                    res.append(chunk["message"]["content"])
                    print(
                        Fore.CYAN + chunk["message"]["content"] + Fore.RESET,
                        end="",
                        flush=True,
                    )
                print("\n")
                model._variables[name] = "".join(res)
            else:
                model._variables[name] = response["message"]["content"]
        return model


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
                if name.startswith("__"):
                    continue
                val = globals_as_dict[name]
                # Ignore functions - we really only want scalars here
                if any(x for x in [callable(val), hasattr(val, "__module__")]):
                    continue
                resolved_names += f"{val}\n"
    return f"{call_content}{resolved_names}"
