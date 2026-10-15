"""rerank 精排：DashScope 文本重排。

参考运维助手（agent_py）llm/rerank.py 的实现与请求格式。
"""

from __future__ import annotations

import math

import httpx

from . import config


def rerank(query: str, documents: list[str], top_n: int = 5) -> list[tuple[int, float]]:
    """
    对候选文档重新打分，返回 [(原下标, 相关度分数)]，按分数降序。
    失败时抛异常，由调用方降级为 RRF 顺序。
    """
    if not documents:
        return []
    top_n = min(top_n, len(documents))
    payload = {
        "model": config.RERANK_MODEL,
        "input": {
            "query": {"text": query},
            "documents": [{"text": d} for d in documents],
        },
        "parameters": {"return_documents": False, "top_n": top_n},
    }
    resp = httpx.post(
        config.RERANK_URL,
        headers={
            "Authorization": f"Bearer {config.DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()["output"]["results"]

    out: list[tuple[int, float]] = []
    seen: set[int] = set()
    for r in results:
        index, score = r.get("index"), r.get("relevance_score")
        # 防御：下标越界/重复、分数非有限数都丢弃
        if not isinstance(index, int) or index in seen or not 0 <= index < len(documents):
            continue
        if not isinstance(score, int | float) or not math.isfinite(float(score)):
            continue
        seen.add(index)
        out.append((index, float(score)))
    out.sort(key=lambda x: -x[1])
    return out
