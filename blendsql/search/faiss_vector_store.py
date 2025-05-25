import os
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import numpy as np
import platformdirs
import typing as t

from colorama import Fore

from blendsql.common.logger import logger
from blendsql.search.searcher import Searcher

ReturnObj = t.TypeVar("ReturnObj")


def maybe_make_dir(dir_path: Path):
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)


def dependable_faiss_import(no_avx2: t.Optional[bool] = None) -> t.Any:
    """
    https://python.langchain.com/v0.2/api_reference/_modules/langchain_community/vectorstores/faiss.html#dependable_faiss_import
    Import faiss if available, otherwise raise error.
    If FAISS_NO_AVX2 environment variable is set, it will be considered
    to load FAISS with no AVX2 optimization.

    Args:
        no_avx2: Load FAISS strictly with no AVX2 optimization
            so that the vectorstore is portable and compatible with other devices.
    """
    if no_avx2 is None and "FAISS_NO_AVX2" in os.environ:
        no_avx2 = bool(os.getenv("FAISS_NO_AVX2"))

    try:
        if no_avx2:
            from faiss import swigfaiss as faiss
        else:
            import faiss
    except ImportError as e:
        raise e from None
    return faiss


@dataclass(kw_only=True)
class FaissVectorStore(Searcher):
    documents: t.List[str] = field()
    # https://github.com/facebookresearch/faiss/wiki/The-index-factory
    factory_str: str = field(default="Flat")
    model_name_or_path: str = field(default="sentence-transformers/all-mpnet-base-v2")
    index_dir: Path = field(
        default=Path(platformdirs.user_cache_dir("blendsql")) / "faiss_vectors"
    )
    return_objs: t.List[ReturnObj] = field(default=None)
    st_encode_kwargs: t.Optional[dict[str, t.Any]] = field(default=None)
    batch_size: t.Optional[int] = field(default=32)

    index: "faiss.Index" = field(init=False)
    embedding_model: "SentenceTransformer" = field(init=False)
    idx_to_return_obj: t.Dict[int, ReturnObj] = field(init=False)
    hashed_documents_str: str = field(init=False)

    def __post_init__(self):
        faiss = dependable_faiss_import()
        import torch
        from numpy.typing import NDArray
        from sentence_transformers import SentenceTransformer

        self.id_to_return_obj = {}
        if self.return_objs is not None:
            # Sort documents, and make sure return_objs are sorted in the same way
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

        # Load SentenceTransformer and any kwargs we need to pass on encode
        self.embedding_model = SentenceTransformer(
            self.model_name_or_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if self.st_encode_kwargs is None:
            self.st_encode_kwargs = {}

        self.embedding_dimension = (
            self.embedding_model.get_sentence_embedding_dimension()
        )

        maybe_make_dir(self.index_dir)

        # Check - do we already have these vectors stored somewhere?
        hasher = hashlib.md5()
        hasher.update(str(sorted(str(self.documents))).encode())
        self.hashed_documents_str = hasher.hexdigest()

        curr_index_path = (
            self.index_dir
            / self.model_name_or_path.replace("/", "_")
            / self.factory_str
            / f"{self.hashed_documents_str}.bin"
        )
        maybe_make_dir(curr_index_path.parent)

        if curr_index_path.is_file():
            logger.debug(
                Fore.MAGENTA + "Loading faiss vectors from cached index..." + Fore.RESET
            )
            self.index = faiss.read_index(str(curr_index_path))
        else:
            logger.debug(Fore.YELLOW + "Creating faiss vectors..." + Fore.RESET)
            self.index = faiss.index_factory(
                self.embedding_model.get_sentence_embedding_dimension(),
                self.factory_str,
            )
            embeddings: NDArray[np.float32] = self.embedding_model.encode(
                self.documents, progress_bar=True
            )
            self.index.add(embeddings)
            faiss.write_index(self.index, str(curr_index_path))

    def __call__(
        self,
        query: t.Union[t.List[str], str],
        k: t.Optional[int] = None,
        scores_only: bool = False,
    ) -> t.List[t.List[str]]:
        is_single_query = isinstance(query, str)
        queries = [query] if is_single_query else query

        # Encode all queries in batch
        query_embeddings = self.embedding_model.encode(
            queries, batch_size=self.batch_size, **self.st_encode_kwargs
        )

        # Reshape if single query to match FAISS expected dimensions
        if is_single_query:
            query_embeddings = query_embeddings.reshape(1, -1)

        # Perform batch search
        distances, indices = self.index.search(query_embeddings, k or self.k)

        if scores_only:
            return (indices, distances)

        # Convert indices to return objects
        results = []
        for batch_indices in indices:
            batch_results = [
                self.idx_to_return_obj[int(i)] for i in batch_indices if i >= 0
            ]
            results.append(batch_results)

        return results
