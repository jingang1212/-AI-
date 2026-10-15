"""向量库封装：一个接口，两种后端（本地 numpy / Milvus）。

为什么不是 Chroma？
实测（Windows + chromadb 1.x 的 Rust 引擎）：连最基本的一次 upsert 都会触发
原生层 access violation，Python 进程直接崩，异常都抓不到。
定位过程：用 `python -X faulthandler scripts/ingest.py` 打出崩溃栈，
看到崩在 chromadb/api/rust.py 的 _upsert / _delete。
→ 结论：本地开发环境不引入不可控的原生依赖，把向量库抽象成接口，
   本地用 numpy 实现（几十万字以内完全够用），生产切 Milvus。

这一层的价值（也是面试可以讲的架构点）：
    rag.py / tools 只调用 add_chunks / search / get_all_chunks 三个函数，
    换向量库只改本文件，业务代码零改动。

Milvus 版本的核心写法（生产用，见文末注释）：
    client = MilvusClient(uri=..., token=...)
    client.search(collection_name="knowledge_chunks", data=[vec],
                  anns_field="vector", limit=top_k,
                  search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                  filter='kb_id == "kb_general"', output_fields=["content", "source"])
    其中 filter 就是「标量过滤」，生产上用来做租户/知识库隔离，必须一开始就设计进来。
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

from . import config
from .embeddings import embed_query, embed_texts

# 向量数据落盘文件。几十万字 ≈ 几千个 chunk，几 MB JSON，读取完全无压力；
# 数据量再大（10 万 chunk 以上）就该换 Milvus，不要在这里做工程化硬扛。
VECTORS_PATH = config.DATA_DIR / "vectors.json"

_DATA: dict[str, dict[str, Any]] | None = None

# 语料版本号：每次写入/删除自增。
# 上层（rag 的 BM25 索引缓存）用它判断「语料变了没有」，
# 没变就直接复用内存里的索引，避免每次提问都重建。
_VERSION = 0


def version() -> int:
    return _VERSION


def _load() -> dict[str, dict[str, Any]]:
    """惰性加载向量库到内存（进程内单例）。"""
    global _DATA
    if _DATA is None:
        if VECTORS_PATH.exists():
            _DATA = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
        else:
            _DATA = {}
    return _DATA


def _save() -> None:
    """原子落盘：先写临时文件再 replace，避免写到一半中断导致数据文件损坏。

    这是「用本地文件当数据库」最容易忽略的细节：
    直接覆盖写，一旦中断，整个知识库就废了。
    """
    VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = VECTORS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(_load(), ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, VECTORS_PATH)  # os.replace 在 Windows 上也是原子覆盖
    global _VERSION
    _VERSION += 1


def _normalize(vector: list[float]) -> list[float]:
    """把向量归一化成长度为 1 的单位向量。

    归一化后余弦相似度就等于点积：检索时少算一次除法，
    而且分数固定在 0~1，前端展示「相关度 0.86」很直观。
    """
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def add_chunks(chunks: list[dict]) -> None:
    """写入/覆盖 chunks。

    chunks: [{"text": str, "metadata": {...}}]
    主键 = f"{document_id}:{chunk_index}"，**同一个 id 再写就是覆盖**：
    这就是「改了文档重新导入是覆盖还是新增」的答案——覆盖，不会产生重复数据。
    """
    if not chunks:
        return
    vectors = embed_texts([c["text"] for c in chunks])  # 一次批量请求，比逐条快得多

    # 维度自检：换 embedding 模型时最容易在这里翻车（Milvus 里会直接建表失败）
    dim = len(vectors[0])
    if dim != config.EMBED_DIM:
        raise ValueError(
            f"向量维度 {dim} 与配置 EMBED_DIM={config.EMBED_DIM} 不一致，"
            "请确认 EMBED_MODEL 与 collection 定义是否配套"
        )

    data = _load()
    for chunk, vector in zip(chunks, vectors):
        meta = chunk["metadata"]
        key = f"{meta.get('document_id')}:{meta.get('chunk_index')}"
        data[key] = {
            "text": chunk["text"],
            "metadata": meta,
            "vector": _normalize(vector),
        }
    _save()


def delete_document(document_id: str) -> int:
    """删除某篇文档的全部 chunk，返回删除条数。

    用途：文档改版后 chunk 数量变少时，清掉多余的历史尾巴，
    否则旧版本残留的片段会继续被检索到（RAG 里很隐蔽的脏数据问题）。
    """
    data = _load()
    stale = [k for k, v in data.items() if v["metadata"].get("document_id") == document_id]
    for k in stale:
        del data[k]
    if stale:
        _save()
    return len(stale)


def search(query: str, top_k: int = config.DEFAULT_TOP_K, where: dict | None = None) -> list[dict]:
    """向量检索：返回带来源与余弦相似度的命中。

    where 是「标量过滤」的本地实现（例如 {"kb_id": "kb_general"}）。
    注意过滤要在**打分之前**做（先缩小候选集再算相似度），
    否则既浪费算力，又可能让不该看到的租户数据进入候选（权限泄漏）。
    """
    data = _load()
    if not data:
        return []

    q = np.asarray(_normalize(embed_query(query)), dtype=np.float32)
    keys: list[str] = []
    matrix: list[list[float]] = []
    for key, item in data.items():
        if where and any(item["metadata"].get(k) != v for k, v in where.items()):
            continue
        keys.append(key)
        matrix.append(item["vector"])
    if not keys:
        return []

    scores = np.asarray(matrix, dtype=np.float32) @ q  # 归一化后点积 = 余弦相似度
    order = np.argsort(-scores)[:top_k]
    return [
        {
            "key": keys[i],
            "text": data[keys[i]]["text"],
            "source": data[keys[i]]["metadata"].get("source"),
            "score": round(float(scores[i]), 4),
        }
        for i in order
    ]


def get_all_chunks() -> list[dict]:
    """取出全部 chunk（供 BM25 建索引）。

    性能说明：这里是全量扫描，几十万字没问题；
    上百万 chunk 时应改用独立关键词索引（Elasticsearch / OpenSearch），
    或只对「向量召回的候选集」做 BM25 重排——面试常问「BM25 怎么扩展」。
    """
    return [
        {
            "key": key,
            "text": item["text"],
            "source": item["metadata"].get("source"),
            # 带上 metadata：BM25 召回也要能按知识库/租户过滤，
            # 否则「向量结果过滤了、关键词结果没过滤」会造成越权泄漏。
            "metadata": item["metadata"],
        }
        for key, item in _load().items()
    ]


def reset() -> None:
    """清空本地向量库（重新全量导入前用）。"""
    global _DATA
    _DATA = {}
    _save()


# ---------------------------------------------------------------------------
# 生产版（Milvus）参考实现，切换时把上面的函数替换成下面的写法即可：
#
# from pymilvus import MilvusClient
#
# class MilvusStore:
#     def __init__(self, uri: str, token: str = ""):
#         self.client = MilvusClient(uri=uri, token=token)   # 阿里云托管版传 uri + token
#
#     def add_chunks(self, chunks):
#         rows = [{"chunk_id": key, "vector": vec, "content": text, **meta} for ...]
#         for i in range(0, len(rows), 10):                  # 分批写，单次别太大
#             self.client.upsert(collection_name="knowledge_chunks", data=rows[i:i+10])
#
#     def search(self, query, top_k, where=None):
#         vec = embed_query(query)
#         expr = " and ".join(f'{k} == "{v}"' for k, v in (where or {}).items()) or None
#         hits = self.client.search(
#             collection_name="knowledge_chunks",
#             data=[vec],
#             anns_field="vector",
#             limit=top_k,
#             search_params={"metric_type": "COSINE", "params": {"ef": 64}},
#             filter=expr,                                   # 标量过滤：租户隔离靠它
#             output_fields=["content", "source"],
#         )
#         return [...]
#
# 建表要点：dim 必须 = config.EMBED_DIM（1024）；索引用 HNSW(M=16, efConstruction=200)；
# 标量字段 kb_id / tenant_id 要建 INVERTED 索引，否则 filter 会退化成全表扫描。
# ---------------------------------------------------------------------------
