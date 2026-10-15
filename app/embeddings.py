"""向量化：DashScope text-embedding-v4（默认 1024 维）。"""

from __future__ import annotations

from langchain_community.embeddings import DashScopeEmbeddings

from . import config

_emb = None


def get_embeddings():
    """单例 embedding 模型。"""
    global _emb
    if _emb is None:
        _emb = DashScopeEmbeddings(
            model=config.EMBED_MODEL,
            dashscope_api_key=config.DASHSCOPE_API_KEY,
        )
    return _emb


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embeddings().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    return get_embeddings().embed_query(text)
