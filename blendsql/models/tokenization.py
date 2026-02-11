from huggingface_hub import hf_hub_download

from blendsql.common.logger import logger

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

from packaging import version

import jinja2
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment

# from PIL.Image import Image


BASIC_TYPES = (int, float, str, bool, Any, type(None), ...)
# Extracts the initial segment of the docstring, containing the function description
description_re = re.compile(r"^(.*?)[\n\s]*(Args:|Returns:|Raises:|\Z)", re.DOTALL)
# Extracts the Args: block from the docstring
args_re = re.compile(r"\n\s*Args:\n\s*(.*?)[\n\s]*(Returns:|Raises:|\Z)", re.DOTALL)
# Splits the Args: block into individual arguments
args_split_re = re.compile(
    r"""
(?:^|\n)  # Match the start of the args block, or a newline
\s*(\w+):\s*  # Capture the argument name and strip spacing
(.*?)\s*  # Capture the argument description, which can span multiple lines, and strip trailing spacing
(?=\n\s*\w+:|\Z)  # Stop when you hit the next argument or the end of the block
""",
    re.DOTALL | re.VERBOSE,
)
# Extracts the Returns: block from the docstring, if present. Note that most chat templates ignore the return type/doc!
returns_re = re.compile(r"\n\s*Returns:\n\s*(.*?)[\n\s]*(Raises:|\Z)", re.DOTALL)


class TypeHintParsingException(Exception):
    """Exception raised for errors in parsing type hints to generate JSON schemas"""


class DocstringParsingException(Exception):
    """Exception raised for errors in parsing docstrings to generate JSON schemas"""


@lru_cache
def _compile_jinja_template(chat_template):
    class AssistantTracker(Extension):
        # This extension is used to track the indices of assistant-generated tokens in the rendered chat
        tags = {"generation"}

        def __init__(self, environment: ImmutableSandboxedEnvironment):
            # The class is only initiated by jinja.
            super().__init__(environment)
            environment.extend(activate_tracker=self.activate_tracker)
            self._rendered_blocks = None
            self._generation_indices = None

        def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.CallBlock:
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
            return jinja2.nodes.CallBlock(
                self.call_method("_generation_support"), [], [], body
            ).set_lineno(lineno)

        @jinja2.pass_eval_context
        def _generation_support(
            self, context: jinja2.nodes.EvalContext, caller: jinja2.runtime.Macro
        ) -> str:
            rv = caller()
            if self.is_active():
                # Only track generation indices if the tracker is active
                start_index = len("".join(self._rendered_blocks))
                end_index = start_index + len(rv)
                self._generation_indices.append((start_index, end_index))
            return rv

        def is_active(self) -> bool:
            return self._rendered_blocks or self._generation_indices

        @contextmanager
        def activate_tracker(
            self, rendered_blocks: list[int], generation_indices: list[int]
        ):
            try:
                if self.is_active():
                    raise ValueError(
                        "AssistantTracker should not be reused before closed"
                    )
                self._rendered_blocks = rendered_blocks
                self._generation_indices = generation_indices

                yield
            finally:
                self._rendered_blocks = None
                self._generation_indices = None

    if version.parse(jinja2.__version__) < version.parse("3.1.0"):
        raise ImportError(
            f"apply_chat_template requires jinja2>=3.1.0 to be installed. Your version is {jinja2.__version__}."
        )

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        # We override the built-in tojson filter because Jinja's default filter escapes HTML characters
        # We also expose some options like custom indents and separators
        return json.dumps(
            x,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    def strftime_now(format):
        return datetime.now().strftime(format)

    jinja_env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[AssistantTracker, jinja2.ext.loopcontrols],
    )
    jinja_env.filters["tojson"] = tojson
    jinja_env.globals["raise_exception"] = raise_exception
    jinja_env.globals["strftime_now"] = strftime_now
    return jinja_env.from_string(chat_template)


def render_jinja_template(
    messages: list[dict[str, str]],
    tools: Optional[list[Union[dict, Callable]]] = None,
    documents: Optional[list[dict[str, str]]] = None,
    chat_template: Optional[str] = None,
    return_assistant_tokens_mask: bool = False,
    continue_final_message: bool = False,
    add_generation_prompt: bool = False,
    **kwargs,
) -> str:
    if return_assistant_tokens_mask and not re.search(
        r"\{\%-?\s*generation\s*-?\%\}", chat_template
    ):
        logger.warning_once(
            "return_assistant_tokens_mask==True but chat template does not contain `{% generation %}` keyword."
        )

    # Compilation function uses a cache to avoid recompiling the same template
    compiled_template = _compile_jinja_template(chat_template)

    if documents is not None:
        for document in documents:
            if not isinstance(document, dict):
                raise TypeError(
                    "Documents should be a list of dicts with 'title' and 'text' keys!"
                )

    continue_final_message_tag = "CONTINUE_FINAL_MESSAGE_TAG "
    if hasattr(messages, "messages"):
        # Indicates it's a Conversation object
        messages = messages.messages
    if continue_final_message:
        messages = deepcopy(messages)
        final_message = messages[-1]["content"]
        if isinstance(final_message, (list, tuple)):
            for content_block in reversed(final_message):
                if "text" in content_block:
                    # Pick the last text block in the message (the first one we hit while iterating in reverse)
                    final_message = content_block["text"]
                    content_block["text"] = (
                        content_block["text"] + continue_final_message_tag
                    )
                    break
            else:
                raise ValueError(
                    "continue_final_message is set but we could not find any text to continue in the final message!"
                )
        else:
            messages[-1]["content"] = (
                messages[-1]["content"] + continue_final_message_tag
            )

    rendered_messages = compiled_template.render(
        messages=messages,
        documents=documents,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    if continue_final_message:
        if (final_message.strip() not in rendered_messages) or (
            continue_final_message_tag.strip() not in rendered_messages
        ):
            raise ValueError(
                "continue_final_message is set but the final message does not appear in the chat after "
                "applying the chat template! This can happen if the chat template deletes portions of "
                "the final message. Please verify the chat template and final message in your chat to "
                "ensure they are compatible."
            )
        tag_loc = rendered_messages.rindex(continue_final_message_tag.strip())
        if (
            rendered_messages[tag_loc : tag_loc + len(continue_final_message_tag)]
            == continue_final_message_tag
        ):
            # The template preserves spacing, so things are simple
            rendered_messages = rendered_messages[:tag_loc]
        else:
            # The message has trailing spacing that was trimmed, so we must be more cautious
            rendered_messages = rendered_messages[:tag_loc].rstrip()
    return rendered_messages


def apply_chat_template(
    repo_id, messages, add_generation_prompt: bool, continue_final_message: bool
):
    with open(
        hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json"), "r"
    ) as f:
        config = json.load(f)
    chat_template = config["chat_template"]
    with open(hf_hub_download(repo_id, filename="special_tokens_map.json"), "r") as f:
        special_tokens_map = json.load(f)
    special_tokens_map = {
        k: v["content"] if isinstance(v, dict) else v
        for k, v in special_tokens_map.items()
    }
    return render_jinja_template(
        messages=messages,
        chat_template=chat_template,
        continue_final_message=continue_final_message,
        add_generation_prompt=add_generation_prompt,
        **special_tokens_map,
    )
