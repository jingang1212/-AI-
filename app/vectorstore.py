"""向量库：Milvus（官方 SDK，嵌入式 Lite 与服务端同一套代码）。

后端选择（按配置）：
1. MILVUS_URI 未配置 → 默认使用 Milvus Lite（本仓库 data/ 下的嵌入式文件）；
2. MILVUS_URI=http://host:19530 → 连接 Milvus 服务端（生产形态，无需改代码）；

Milvus 初始化失败会直接抛异常，不做任何降级——静默降级会掩盖根因，
让「服务在跑但搜不到数据」这类问题难以排查。

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

import math
import sys

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
    """Milvus 实现：嵌入式（Lite）与服务端共用一套 API。

    collection 参数允许测试/工具使用独立集合，避免与正式知识库互相污染。
    """

    def __init__(self, collection: str | None = None) -> None:
        from pymilvus import MilvusClient

        self.collection = collection or config.MILVUS_COLLECTION
        # Lite：uri 传本地文件路径；服务端：传 http(s) 地址。token 用于托管版鉴权。
        self.client = MilvusClient(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN or None)
        self._ensure_collection()

    @staticmethod
    def drop_collection_safe(collection: str) -> None:
        """用原始客户端删集合：兼容旧 schema 无法通过体检的场景（升级/测试清理）。"""
        from pymilvus import MilvusClient

        client = MilvusClient(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN or None)
        if client.has_collection(collection):
            client.drop_collection(collection)

    # ---- 建表 ---------------------------------------------------------------

    def _ensure_collection(self) -> None:
        from pymilvus import DataType

        if not self.client.has_collection(self.collection):
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
            schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
            schema.add_field("source", DataType.VARCHAR, max_length=256)
            schema.add_field("document_id", DataType.VARCHAR, max_length=64)
            schema.add_field("chunk_index", DataType.INT64)
            schema.add_field("strategy", DataType.VARCHAR, max_length=32)
            schema.add_field("heading_path", DataType.VARCHAR, max_length=512)
            schema.add_field("start_index", DataType.INT64)
            schema.add_field("end_index", DataType.INT64)
            schema.add_field("parent_id", DataType.VARCHAR, max_length=128)
            schema.add_field("parent_text", DataType.VARCHAR, max_length=65535)

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
            index_params.add_index(field_name="tenant_id", index_type="INVERTED")
            self.client.create_collection(
                collection_name=self.collection, schema=schema, index_params=index_params
            )
            # Milvus 服务端语义：集合要先 load 到内存才能 search/query。
            # 新集合是 released 状态，不 load 会报 code=101。
            self.client.load_collection(self.collection)
            return

        # 已存在：确保加载（重复 load 幂等）。Lite 模式跳过也行，统一处理最稳。
        self.client.load_collection(self.collection)
        # 已存在的 collection 做 schema 体检：
        # 1) 字段齐全性——升级 schema 后旧集合缺字段，必须重建；
        # 2) 向量维度——换 embedding 模型没重建表时，错误会以诡异形态出现。
        fields = self.client.describe_collection(self.collection).get("fields", [])
        field_names = {f.get("name") for f in fields}
        required = {
            "chunk_id", "vector", "content", "kb_id", "tenant_id", "source",
            "document_id", "chunk_index", "strategy", "heading_path", "start_index", "end_index",
            "parent_id", "parent_text",
        }
        missing = required - field_names
        if missing:
            raise RuntimeError(
                f"Milvus collection 缺少字段 {sorted(missing)}：schema 已升级，"
                "请先停掉服务，执行 python scripts/ingest.py --reset 重新入库，再重启服务"
            )
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
        self.client.upsert(collection_name=self.collection, data=rows)

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
                    "tenant_id": meta.get("tenant_id", "default"),
                    "source": meta.get("source", ""),
                    "document_id": meta.get("document_id", ""),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                    "strategy": meta.get("strategy", ""),
                    "heading_path": (meta.get("heading_path") or "")[:512],
                    "start_index": int(meta.get("start", 0)),
                    "end_index": int(meta.get("end", 0)),
                    "parent_id": meta.get("parent_id", ""),
                    "parent_text": meta.get("parent_text", ""),
                }
            )
        self._insert_rows(rows)

    def delete_document(self, document_id: str) -> int:
        """按 document_id 删除全部 chunk（改版重新导入前清旧尾巴）。"""
        deleted = self.client.delete(
            collection_name=self.collection,
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
            collection_name=self.collection,
            data=[_normalize(embed_query(query))],
            limit=top_k,
            filter=self._expr(where),
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            # 必须显式包含主键 chunk_id：自定义主键不会被自动返回，
            # 漏了它 entity 里没有 key，RAG 层合并时会把所有命中塌成同一个空 key
            output_fields=[
                "chunk_id", "content", "source", "kb_id", "document_id",
                "chunk_index", "heading_path", "start_index", "end_index",
                "parent_id", "parent_text",
            ],
        )
        return [
            {
                "key": h["entity"].get("chunk_id", ""),
                "text": h["entity"].get("content", ""),
                "source": h["entity"].get("source"),
                "score": round(float(h["distance"]), 4),  # COSINE 度量下 distance 就是相似度
                "kb_id": h["entity"].get("kb_id"),
                "document_id": h["entity"].get("document_id"),
                "chunk_index": h["entity"].get("chunk_index"),
                "heading_path": h["entity"].get("heading_path", ""),
                "start": h["entity"].get("start_index", 0),
                "end": h["entity"].get("end_index", 0),
                "parent_id": h["entity"].get("parent_id", ""),
                "parent_text": h["entity"].get("parent_text", ""),
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
                collection_name=self.collection,
                filter="",
                output_fields=[
                    "chunk_id", "content", "source", "kb_id", "document_id",
                    "heading_path", "start_index", "end_index", "parent_id", "parent_text",
                ],
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
                    "metadata": {
                        "kb_id": r.get("kb_id"),
                        "document_id": r.get("document_id"),
                        "heading_path": r.get("heading_path", ""),
                        "start": r.get("start_index", 0),
                        "end": r.get("end_index", 0),
                        "parent_id": r.get("parent_id", ""),
                        "parent_text": r.get("parent_text", ""),
                    },
                }
                for r in page
            )
            if len(page) < page_size:
                break
            offset += len(page)
        return out

    def reset(self) -> None:
        self.client.drop_collection(self.collection)
        self._ensure_collection()


# ---------------------------------------------------------------------------
# 模块级 API（rag 层只依赖这几个函数）
# ---------------------------------------------------------------------------

_store: MilvusStore | None = None
_version = 0


def _get_store() -> MilvusStore:
    """进程内单例。初始化失败直接抛异常，不降级。"""
    global _store
    if _store is None:
        _store = MilvusStore()
        print(f"✅ 向量后端：Milvus（{config.MILVUS_URI}）", file=sys.stderr)
    return _store


def _bump() -> None:
    """语料版本号自增：rag 的 BM25 索引缓存靠它判断是否需要重建。"""
    global _version
    _version += 1


def backend_name() -> str:
    return "milvus"


def add_chunks(chunks: list[dict]) -> None:
    _get_store().add_chunks(chunks)
    _bump()


def delete_document(document_id: str) -> int:
    n = _get_store().delete_document(document_id)
    _bump()
    return n


def search(query: str, top_k: int = config.DEFAULT_TOP_K, where: dict | None = None) -> list[dict]:
    return _get_store().search(query, top_k, where)


def get_all_chunks() -> list[dict]:
    return _get_store().get_all_chunks()


def reset() -> None:
    """清空并重建知识库。

    用原始客户端直接删集合：旧集合可能缺新 schema 字段（升级场景），
    若先走 _get_store() 的 schema 体检会直接报错、连删都删不掉。
    """
    from pymilvus import MilvusClient

    global _store
    _store = None  # 旧单例失效，重建后按新 schema 初始化
    client = MilvusClient(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN or None)
    if client.has_collection(config.MILVUS_COLLECTION):
        client.drop_collection(config.MILVUS_COLLECTION)
    _get_store()
    _bump()


def version() -> int:
    return _version
