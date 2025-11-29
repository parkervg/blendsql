from abc import abstractmethod
from dataclasses import dataclass, field


@dataclass
class Searcher:
    k: int | None = field(default=1)

    @abstractmethod
    def __call__(self, query: str | list[str], k: str | None = None) -> list[list[str]]:
        ...
