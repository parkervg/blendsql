import os
import argparse
from tabulate import tabulate

from blendsql import blend, init_secrets
from blendsql.db import SQLiteDBConnector
from blendsql.ingredients.builtin import LLMQA, LLMMap, DT


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", nargs="?")
    parser.add_argument("--secrets_path", default="./secrets.json")
    args = parser.parse_args()

    init_secrets(args.secrets_path)
    db = SQLiteDBConnector(db_path=args.db_path)
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
        smoothie = blend(
            query=text, db=db, ingredients={LLMQA, LLMMap, DT}, verbose=False
        )
        print()
        print(
            tabulate(
                smoothie.df.iloc[:10],
                headers="keys",
                showindex="never",
                tablefmt="orgtbl",
            )
        )
        print()
