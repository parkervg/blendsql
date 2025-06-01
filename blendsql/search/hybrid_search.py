import typing as t
from dataclasses import dataclass, field
from colorama import Fore

from blendsql.common.logger import logger
from blendsql.search.faiss_vector_store import FaissVectorStore


@dataclass(kw_only=True)
class HybridSearch(FaissVectorStore):
    normalization: bool = field(default=True)
    bm25_weight: float = field(default=0.5)

    bm25_method: str = field(
        default="lucene"
    )  # By default, bm25s uses method="lucene", which is Lucene's BM25 implementation (exact version).
    bm25_retriever: "bm25s.BM25" = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        import bm25s

        if self.bm25_weight > 0.0:
            curr_index_dir = (
                self.index_dir / self.bm25_method / self.hashed_documents_str
            )
            if curr_index_dir.is_dir():
                logger.debug(
                    Fore.MAGENTA
                    + "Loading bm25 index from cached index..."
                    + Fore.RESET
                )
                # Load existing indices
                self.bm25_retriever = bm25s.BM25.load(curr_index_dir)
            else:
                logger.debug(Fore.YELLOW + "Creating bm25 index..." + Fore.RESET)
                curr_index_dir.mkdir(parents=True)
                corpus_tokens = bm25s.tokenize(self.documents, stopwords="en")
                self.bm25_retriever = bm25s.BM25(method=self.bm25_method)
                self.bm25_retriever.index(corpus_tokens)
                self.bm25_retriever.save(str(curr_index_dir))

    def __call__(
        self, query: t.Union[t.List[str], str], k: t.Optional[int] = None
    ) -> t.List[t.List[str]]:
        """Adapted from https://github.com/castorini/pyserini/blob/7ed83698298139efdfd62b6893d673aa367b4ac8/pyserini/search/hybrid/_searcher.py"""
        import bm25s
        import numpy as np

        if self.bm25_weight == 0.0:
            return super().__call__(query=query, k=k)

        use_k = min(100, len(self.documents))
        is_single_query = isinstance(query, str)
        queries = [query] if is_single_query else query

        faiss_indices, faiss_scores = super().__call__(
            query=queries, k=use_k, scores_only=True
        )
        faiss_scores = 1 - faiss_scores  # Since faiss scores are a distance
        bm25_indices, bm25_scores = self.bm25_retriever.retrieve(
            bm25s.tokenize(queries, stopwords="en"), k=use_k
        )

        assert faiss_indices.shape == bm25_indices.shape

        final_indices = []
        for query_idx in range(faiss_indices.shape[0]):
            curr_scores = []
            # Use NumPy's np.intersect1d for fast intersection
            document_intersection = np.intersect1d(
                faiss_indices[query_idx, :], bm25_indices[query_idx, :]
            )
            for doc_idx in document_intersection:
                # In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
                # i.e, a way to do arr.index(val) on a numpy array
                faiss_score = faiss_scores[
                    query_idx, np.argmax(faiss_indices[query_idx, :] == doc_idx)
                ]
                bm25_score = bm25_scores[
                    query_idx, np.argmax(bm25_indices[query_idx, :] == doc_idx)
                ]
                if self.normalization:
                    min_faiss, max_faiss = faiss_scores.min(), faiss_scores.max()
                    faiss_score = (faiss_score - (min_faiss + max_faiss) / 2) / (
                        max_faiss - min_faiss
                    )
                    min_bm25, max_bm25 = bm25_scores.min(), bm25_scores.max()
                    bm25_score = (bm25_score - (min_bm25 + max_bm25) / 2) / (
                        max_bm25 - min_bm25
                    )
                score = (bm25_score * self.bm25_weight) + (
                    faiss_score * (1 - self.bm25_weight)
                )
                curr_scores.append(score)
            curr_indices = []
            for i in np.argsort(-np.array(curr_scores))[: k or self.k]:
                curr_indices.append(document_intersection[i])
            final_indices.append(curr_indices)

        results = []
        for batch_indices in final_indices:
            batch_results = [
                self.idx_to_return_obj[int(i)] for i in batch_indices if i >= 0
            ]
            results.append(batch_results)

        return results
