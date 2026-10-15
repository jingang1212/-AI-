"""向量库：先用 Chroma 跑通逻辑，后续可切 Milvus（接口不变）。"""

from __future__ import annotations

from langchain_community.vectorstores import Chroma

from . import config
from .embeddings import get_embeddings

_vs = None


def get_store() -> Chroma:
    """单例 Chroma，cosine 度量。"""
    global _vs
    if _vs is None:
        _vs = Chroma(
            collection_name="knowledge",
            embedding_function=get_embeddings(),
            persist_directory=str(config.CHROMA_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _vs


def add_chunks(chunks: list[dict]) -> None:
    """chunks: [{text, metadata}]。"""
    get_store().add_texts(
        texts=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def search(query: str, top_k: int = config.DEFAULT_TOP_K) -> list[dict]:
    """向量检索，返回带分数和出处的命中。"""
    results = get_store().similarity_search_with_score(query, k=top_k)
    return [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source"),
            "score": round(score, 4),
        }
        for doc, score in results
    ]
