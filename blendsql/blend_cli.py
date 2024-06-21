import os
import argparse
import importlib

from blendsql import blend
from blendsql.db import SQLite
from blendsql.db.utils import truncate_df_content
from blendsql.utils import tabulate
from blendsql.models import (
    OpenaiLLM,
    TransformersLLM,
    AzureOpenaiLLM,
    LlamaCppLLM,
    OllamaLLM,
)
from blendsql.ingredients.builtin import LLMQA, LLMMap, LLMJoin

_has_readline = importlib.util.find_spec("readline") is not None

MODEL_TYPE_TO_CLASS = {
    "openai": OpenaiLLM,
    "azure_openai": AzureOpenaiLLM,
    "llama_cpp": LlamaCppLLM,
    "transformers": TransformersLLM,
    "ollama": OllamaLLM,
}


def print_msg_box(msg, indent=1, width=None, title=None):
    """Print message-box with optional title."""
    lines = msg.split("\n")
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f"║{space}{title:<{width}}{space}║\n"  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += "".join([f"║{space}{line:<{width}}{space}║\n" for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    print(box)


def cls():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    if _has_readline:
        import readline

        _ = readline
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", nargs="?", help="Database path")
    parser.add_argument(
        "model_type",
        nargs="?",
        default="openai",
        choices=list(MODEL_TYPE_TO_CLASS.keys()),
        help="Model type, for the Blender to use in executing the BlendSQL query.",
    )
    parser.add_argument(
        "model_name_or_path",
        nargs="?",
        default="gpt-3.5-turbo",
        help="Model identifier to pass to the selected model_type class.",
    )
    parser.add_argument("-v", action="store_true", help="Flag to run in verbose mode.")
    args = parser.parse_args()

    db = SQLite(db_path=args.db_path)
    print_msg_box(f"Beginning BlendSQL session with '{args.db_path}'...")
    print()
    while True:
        lines = []
        while True:
            line = input(">>> ")
            if line:
                lines.append(line)
            else:
                break
        text = "\n".join(lines)
        if text.replace("\n", "").strip() == "clear":
            cls()
            print_msg_box("Beginning BlendSQL session...")
            continue
        try:
            smoothie = blend(
                query=text,
                db=db,
                ingredients={LLMQA, LLMMap, LLMJoin},
                default_model=MODEL_TYPE_TO_CLASS.get(args.model_type)(
                    args.model_name_or_path
                ),
                infer_gen_constraints=True,
                verbose=args.v,
            )
            print()
            print(tabulate(truncate_df_content(smoothie.df, 50)))
        except Exception as error:
            print(error)
