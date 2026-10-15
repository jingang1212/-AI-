"""向量库：Milvus 为主（官方 SDK，嵌入式/服务端同一套代码），numpy 仅作应急兜底。

后端选择（按优先级）：
1. MILVUS_URI 未配置 → 默认使用 Milvus Lite（本仓库 data/ 下的嵌入式文件）；
2. MILVUS_URI=http://host:19530 → 连接 Milvus 服务端（生产形态，无需改代码）；
3. Milvus 初始化失败（缺依赖/环境）→ 降级 NumpyStore 并打印警告——
   这只是一个「不让服务崩」的应急开关，不是推荐路径。

Milvus 侧的关键设计（对应 docs/技术方案.md §3.2 / §4.4）：
- 显式 schema：字符串主键 chunk_id + 1024 维 FLOAT_VECTOR + 标量字段；
- HNSW 索引（M=16, efConstruction=200）+ COSINE 度量，检索 ef=64；
- kb_id / document_id 建 INVERTED 索引：标量过滤走索引而不是全表扫描；
- tenant/kb 隔离用 `filter` 表达式在打分前过滤。

官方文档：
- MilvusClient：https://milvus.io/docs/using_milvusclient.md
- 标量过滤：https://milvus.io/docs/filtered-search.md
- Milvus Lite：https://milvus.io/docs/milvus_lite.md
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

import numpy as np

from . import config
from .embeddings import embed_query, embed_texts

# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _normalize(vector: list[float]) -> list[float]:
    """L2 归一化：归一化后点积 = 余弦相似度，分数固定 0~1。"""
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


# ---------------------------------------------------------------------------
# Milvus 后端
# ---------------------------------------------------------------------------


class MilvusStore:
    """Milvus 实现：嵌入式（Lite）与服务端共用一套 API。"""

    def __init__(self) -> None:
        from pymilvus import MilvusClient

        # Lite：uri 传本地文件路径；服务端：传 http(s) 地址。token 用于托管版鉴权。
        self.client = MilvusClient(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN or None)
        self._ensure_collection()

    # ---- 建表 ---------------------------------------------------------------

    def _ensure_collection(self) -> None:
        from pymilvus import DataType

        if not self.client.has_collection(config.MILVUS_COLLECTION):
            # 显式 schema。两个实测坑（写进审查记录）：
            # 1) MilvusClient 的快速建表（dimension=...）默认把主键推断成 INT64，
            #    传字符串主键会报 DataNotMatchException；
            # 2) enable_dynamic_field=False 时，insert 必须带上 schema 声明的每个字段，
            #    少一个都会报 "Insert missed an field"。
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=config.EMBED_DIM)
            schema.add_field("content", DataType.VARCHAR, max_length=65535)
            schema.add_field("kb_id", DataType.VARCHAR, max_length=64)
            schema.add_field("source", DataType.VARCHAR, max_length=256)
            schema.add_field("document_id", DataType.VARCHAR, max_length=64)
            schema.add_field("chunk_index", DataType.INT64)
            schema.add_field("strategy", DataType.VARCHAR, max_length=32)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            # 标量过滤字段建倒排索引：没有它 filter 会退化成暴力扫描
            index_params.add_index(field_name="kb_id", index_type="INVERTED")
            index_params.add_index(field_name="document_id", index_type="INVERTED")
            self.client.create_collection(
                collection_name=config.MILVUS_COLLECTION, schema=schema, index_params=index_params
            )
            return

        # 已存在的 collection 做维度体检：换 embedding 模型没重建表时，
        # 错误会以「向量维度不匹配」的诡异形态出现，这里提前拦住。
        fields = self.client.describe_collection(config.MILVUS_COLLECTION).get("fields", [])
        for field in fields:
            if field.get("type") == 101 and str(field.get("params", {}).get("dim")) != str(
                config.EMBED_DIM
            ):
                raise ValueError(
                    f"Milvus collection 维度 {field.get('params', {}).get('dim')} "
                    f"与 EMBED_DIM={config.EMBED_DIM} 不一致：换 embedding 模型后必须重建 collection"
                )

    # ---- 写入 ---------------------------------------------------------------

    def _insert_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        # upsert：主键相同即覆盖，对应「同名文档重新导入 = 覆盖」语义
        self.client.upsert(collection_name=config.MILVUS_COLLECTION, data=rows)

    def add_chunks(self, chunks: list[dict]) -> None:
        """chunks: [{"text", "metadata"}] → 向量化 + upsert。"""
        if not chunks:
            return
        vectors = embed_texts([c["text"] for c in chunks])
        if len(vectors[0]) != config.EMBED_DIM:
            raise ValueError(
                f"向量维度 {len(vectors[0])} 与 EMBED_DIM={config.EMBED_DIM} 不一致，请检查 EMBED_MODEL"
            )
        rows = []
        for chunk, vector in zip(chunks, vectors):
            meta = chunk["metadata"]
            rows.append(
                {
                    "chunk_id": f"{meta.get('document_id')}:{meta.get('chunk_index')}",
                    "vector": _normalize(vector),
                    "content": chunk["text"],
                    "kb_id": meta.get("kb_id", "kb_general"),
                    "source": meta.get("source", ""),
                    "document_id": meta.get("document_id", ""),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                    "strategy": meta.get("strategy", ""),
                }
            )
        self._insert_rows(rows)

    def delete_document(self, document_id: str) -> int:
        """按 document_id 删除全部 chunk（改版重新导入前清旧尾巴）。"""
        deleted = self.client.delete(
            collection_name=config.MILVUS_COLLECTION,
            filter=f'document_id == "{document_id}"',
        )
        return int(deleted.get("delete_count", 0)) if isinstance(deleted, dict) else 0

    # ---- 检索 ---------------------------------------------------------------

    @staticmethod
    def _expr(where: dict | None) -> str | None:
        if not where:
            return None
        # 标量值都是安全字符串（kb_id 等），拼接表达式的注入面仅限字段名，由调用方白名单控制
        return " and ".join(f'{k} == "{v}"' for k, v in where.items())

    def search(self, query: str, top_k: int = config.DEFAULT_TOP_K, where: dict | None = None) -> list[dict]:
        """向量检索：embed(query) → Milvus HNSW 检索 + 标量过滤。"""
        hits = self.client.search(
            collection_name=config.MILVUS_COLLECTION,
            data=[_normalize(embed_query(query))],
            limit=top_k,
            filter=self._expr(where),
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            output_fields=["content", "source", "kb_id"],
        )
        return [
            {
                "key": h["entity"].get("chunk_id", ""),
                "text": h["entity"].get("content", ""),
                "source": h["entity"].get("source"),
                "score": round(float(h["distance"]), 4),  # COSINE 度量下 distance 就是相似度
            }
            for h in (hits[0] if hits else [])
        ]

    def get_all_chunks(self) -> list[dict]:
        """全量 chunk（供 BM25 建索引）。

        Milvus 单次 query 上限 16384 行，按 offset 分页拿全。
        语料到几十万字（万级 chunk）没问题；再大就该把关键词检索独立到
        Elasticsearch / OpenSearch，而不是全量拉回内存（文档里已注明）。
        """
        page_size = 5000
        offset = 0
        out: list[dict] = []
        while True:
            page = self.client.query(
                collection_name=config.MILVUS_COLLECTION,
                filter="",
                output_fields=["chunk_id", "content", "source", "kb_id", "document_id"],
                limit=page_size,
                offset=offset,
            )
            if not page:
                break
            out.extend(
                {
                    "key": r.get("chunk_id"),
                    "text": r.get("content", ""),
                    "source": r.get("source"),
                    "metadata": {"kb_id": r.get("kb_id"), "document_id": r.get("document_id")},
                }
                for r in page
            )
            if len(page) < page_size:
                break
            offset += len(page)
        return out

    def reset(self) -> None:
        self.client.drop_collection(config.MILVUS_COLLECTION)
        self._ensure_collection()


# ---------------------------------------------------------------------------
# numpy 应急兜底（不用 Milvus 时的降级路径，非推荐）
# ---------------------------------------------------------------------------


class NumpyStore:
    """本地 numpy 实现：无任何服务依赖，但缺少索引、并发与标量过滤的工程能力。"""

    def __init__(self) -> None:
        self.path = config.DATA_DIR / "vectors.json"
        self._data: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._data is None:
            self._data = (
                json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
            )
        return self._data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._load(), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def add_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        vectors = embed_texts([c["text"] for c in chunks])
        data = self._load()
        for chunk, vector in zip(chunks, vectors):
            meta = chunk["metadata"]
            key = f"{meta.get('document_id')}:{meta.get('chunk_index')}"
            data[key] = {"text": chunk["text"], "metadata": meta, "vector": _normalize(vector)}
        self._save()

    def delete_document(self, document_id: str) -> int:
        data = self._load()
        stale = [k for k, v in data.items() if v["metadata"].get("document_id") == document_id]
        for k in stale:
            del data[k]
        if stale:
            self._save()
        return len(stale)

    def search(self, query: str, top_k: int = config.DEFAULT_TOP_K, where: dict | None = None) -> list[dict]:
        data = self._load()
        if not data:
            return []
        q = np.asarray(_normalize(embed_query(query)), dtype=np.float32)
        keys, matrix = [], []
        for key, item in data.items():
            if where and any(item["metadata"].get(k) != v for k, v in where.items()):
                continue
            keys.append(key)
            matrix.append(item["vector"])
        if not keys:
            return []
        scores = np.asarray(matrix, dtype=np.float32) @ q
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

    def get_all_chunks(self) -> list[dict]:
        return [
            {"key": key, "text": item["text"], "source": item["metadata"].get("source"),
             "metadata": item["metadata"]}
            for key, item in self._load().items()
        ]

    def reset(self) -> None:
        self._data = {}
        self._save()


# ---------------------------------------------------------------------------
# 后端选择与模块级 API（rag 层只依赖这几个函数）
# ---------------------------------------------------------------------------

_backend = None
_version = 0


def _bump() -> None:
    """语料版本号自增：rag 的 BM25 索引缓存靠它判断是否需要重建。"""
    global _version
    _version += 1


def _init_backend():
    global _backend
    if _backend is not None:
        return _backend
    try:
        _backend = MilvusStore()
        print(f"✅ 向量后端：Milvus（{config.MILVUS_URI}）", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        _backend = NumpyStore()
        print(
            f"⚠️ Milvus 初始化失败（{type(exc).__name__}: {str(exc)[:120]}），"
            "降级为本地 numpy 向量库（仅应急，不具备生产能力）",
            file=sys.stderr,
        )
    return _backend


def backend_name() -> str:
    return type(_init_backend()).__name__.replace("Store", "").lower()


def add_chunks(chunks: list[dict]) -> None:
    _init_backend().add_chunks(chunks)
    _bump()


def delete_document(document_id: str) -> int:
    n = _init_backend().delete_document(document_id)
    _bump()
    return n


def search(query: str, top_k: int = config.DEFAULT_TOP_K, where: dict | None = None) -> list[dict]:
    return _init_backend().search(query, top_k, where)


def get_all_chunks() -> list[dict]:
    return _init_backend().get_all_chunks()


def reset() -> None:
    _init_backend().reset()
    _bump()


def version() -> int:
    return _version
