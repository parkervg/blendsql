from dataclasses import dataclass
from pathlib import Path
from attr import attrs, attrib
from typing import List


@attrs
class Examples:
    data: str = attrib()

    split_data: List[str] = attrib(init=False)

    def __attrs_post_init__(self):
        self.data = self.data.strip()
        self.split_data: list = self.data.split("---")

    def __getitem__(self, subscript):
        newline = (
            "\n\n"
            if (isinstance(subscript, int) and subscript == 0)
            or (isinstance(subscript, slice) and subscript.start == 0)
            else ""
        )
        return "Examples:" + newline + "---".join(self.split_data[subscript])

    def __repr__(self):
        return "Examples:\n" + self.data

    def __str__(self):
        return "Examples:\n" + self.data


@dataclass
class FewShot:
    hybridqa = Examples(open(Path(__file__).parent / "./few_shot/hybridqa.txt").read())
