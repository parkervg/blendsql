from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import numpy as np
import platformdirs
import typing as t
import faiss
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from colorama import Fore

from blendsql.common.logger import logger

ReturnObj = t.TypeVar("ReturnObj")


def maybe_make_dir(dir_path: Path):
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)


@dataclass
class FaissVectorStore:
    documents: t.List[str] = field()
    # https://github.com/facebookresearch/faiss/wiki/The-index-factory
    factory_str: str = field(default="Flat")
    model_name_or_path: str = field(default="sentence-transformers/all-mpnet-base-v2")
    index_dir: Path = field(
        default=Path(platformdirs.user_cache_dir("blendsql")) / "faiss_vectors"
    )
    return_objs: t.List[ReturnObj] = field(default=None)

    index: faiss.Index = field(init=False)
    embedding_model: SentenceTransformer = field(init=False)
    idx_to_return_obj: t.Dict[int, ReturnObj] = field(init=False)

    def __post_init__(self):
        self.id_to_return_obj = {}
        if self.return_objs is not None:
            # print(self.return_objs)
            # print(self.documents)
            self.return_objs, self.documents = zip(
                *sorted(
                    [(r, d) for r, d in zip(self.return_objs, self.documents)],
                    key=lambda x: x[1],
                )
            )
            assert len(self.return_objs) == len(self.documents)
            self.idx_to_return_obj = {
                idx: return_obj for idx, return_obj in enumerate(self.return_objs)
            }
        else:
            self.documents = sorted(self.documents)
            self.idx_to_return_obj = {i: doc for i, doc in enumerate(self.documents)}
        self.embedding_model = SentenceTransformer(
            self.model_name_or_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.embedding_dimension = (
            self.embedding_model.get_sentence_embedding_dimension()
        )

        maybe_make_dir(self.index_dir)

        # Check - do we already have these vectors stored somewhere?
        hasher = hashlib.md5()
        hasher.update(str(sorted(str(self.documents))).encode())
        hashed = hasher.hexdigest()

        curr_index_path = (
            self.index_dir
            / self.model_name_or_path.replace("/", "_")
            / self.factory_str
            / f"{hashed}.bin"
        )
        maybe_make_dir(curr_index_path.parent)

        if curr_index_path.is_file():
            logger.debug(
                Fore.CYAN + "Loading faiss vectors from cached index..." + Fore.RESET
            )
            self.index = faiss.read_index(str(curr_index_path))
        else:
            logger.debug(Fore.CYAN + "Creating faiss vectors..." + Fore.RESET)
            self.index = faiss.index_factory(
                self.embedding_model.get_sentence_embedding_dimension(),
                self.factory_str,
            )
            embeddings: NDArray[np.float32] = self.embedding_model.encode(
                self.documents, progress_bar=True
            )
            self.index.add(embeddings)
            faiss.write_index(self.index, str(curr_index_path))

    def __call__(self, query: str, k: int = 3) -> t.List[str]:
        _, indices = self.index.search(
            self.embedding_model.encode(query).reshape(1, -1), k
        )
        return [self.idx_to_return_obj[i] for i in indices[0, :]]


if __name__ == "__main__":
    vs = FaissVectorStore(
        docs=["This is a story about golf and soccer", "Politics are messy"],
        # return_objs=["a", "B"]
    )
    print(vs("I like basketball", 1))
