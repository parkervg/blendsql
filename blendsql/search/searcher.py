import typing as t
from abc import abstractmethod
from dataclasses import dataclass, field


@dataclass
class Searcher:
    documents: t.List[str] = field()

    @abstractmethod
    def __call__(
        self, query: t.Union[str, t.List[str]], k: t.Optional[int] = None
    ) -> t.List[t.List[str]]:
        ...
