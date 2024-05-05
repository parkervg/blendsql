from dataclasses import dataclass
from pathlib import Path


@dataclass
class FewShot:
    hybridqa = open(Path(__file__).parent / "./few_shot/hybridqa.txt").read()
