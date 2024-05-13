import os
import argparse
import importlib
import json

from blendsql import blend
from blendsql.db import SQLite
from blendsql.utils import tabulate
from blendsql.models import LlamaCppLLM
from blendsql.ingredients.builtin import LLMQA, LLMMap, LLMJoin, DT

_has_readline = importlib.util.find_spec("readline") is not None


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
    parser.add_argument("db_path", nargs="?")
    parser.add_argument("secrets_path", nargs="?", default="./secrets.json")
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
                ingredients={LLMQA, LLMMap, LLMJoin, DT},
                blender=LlamaCppLLM(
                    "./lark-constrained-parsing/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
                ),
                infer_gen_constraints=True,
                verbose=True,
            )
            print()
            print(tabulate(smoothie.df.iloc[:10]))
            print()
            print(json.dumps(smoothie.meta.prompts, indent=4))
        except Exception as error:
            print(error)


"""
SELECT "common name" AS 'State Flower' FROM w 
WHERE state = {{
    LLMQA(
        'Which is the smallest state by area?',
        (SELECT title, content FROM documents WHERE documents MATCH 'smallest OR state OR area' LIMIT 3),
        options='w::state'
    )
}}

SELECT Symbol, Description, Quantity FROM portfolio WHERE {{LLMMap('Do they manufacture cell phones?', 'portfolio::Description')}} = TRUE AND portfolio.Symbol in (SELECT Symbol FROM constituents WHERE Sector = 'Information Technology')
"""
