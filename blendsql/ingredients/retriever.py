import os
from attr import attrs, attrib, Factory
from typing import List, Dict, TypeVar, Union
import haystack.document_stores.types
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersTextEmbedder,
    SentenceTransformersDocumentEmbedder,
)
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack import Document, Pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ReturnObj = TypeVar("ReturnObj")


@attrs
class Retriever:
    documents: List[str] = attrib()
    return_objs: List[ReturnObj] = attrib(default=None)
    document_store: haystack.document_stores.types.DocumentStore = attrib(
        default=Factory(
            lambda: InMemoryDocumentStore(embedding_similarity_function="dot_product")
        )
    )
    document_embedder: SentenceTransformersDocumentEmbedder = attrib(
        default=Factory(
            lambda: SentenceTransformersDocumentEmbedder(
                model="TaylorAI/gte-tiny", progress_bar=False
            )
        )
    )
    text_embedder: SentenceTransformersTextEmbedder = attrib(
        default=Factory(
            lambda: SentenceTransformersTextEmbedder(
                model="TaylorAI/gte-tiny", progress_bar=False
            )
        )
    )

    id_to_return_obj: Dict[str, ReturnObj] = attrib(init=False)
    query_pipeline: Pipeline = attrib(init=False)

    def __attrs_post_init__(self):
        self.id_to_return_obj = {}
        # Call each example's to_string() method and adds to index
        documents = [Document(content=doc) for doc in self.documents]
        if self.return_objs:
            assert len(documents) == len(self.return_objs)
            self.id_to_return_obj = {
                doc.id: obj for doc, obj in zip(documents, self.return_objs)
            }
        self.document_embedder.warm_up()
        documents_with_embeddings = self.document_embedder.run(documents)["documents"]
        self.document_store.write_documents(documents_with_embeddings)

        self.query_pipeline = Pipeline()
        self.query_pipeline.add_component("text_embedder", self.text_embedder)
        self.query_pipeline.add_component(
            "retriever", InMemoryEmbeddingRetriever(document_store=self.document_store)
        )
        self.query_pipeline.connect(
            "text_embedder.embedding", "retriever.query_embedding"
        )

    def retrieve_top_k(self, query: str, k: int) -> List[Union[str, ReturnObj]]:
        result = self.query_pipeline.run(
            {"text_embedder": {"text": query}, "retriever": {"top_k": k}}
        )
        return [
            self.id_to_return_obj.get(doc.id, doc.content)
            for doc in result["retriever"]["documents"]
        ]


if __name__ == "__main__":
    from blendsql.ingredients.builtin.map.main import DEFAULT_MAP_FEW_SHOT

    r = Retriever(examples=DEFAULT_MAP_FEW_SHOT)
    print(r.retrieve_top_k("test", k=1))
