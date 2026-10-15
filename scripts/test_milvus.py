"""Milvus 向量库机制测试（纯本地，不调模型、不需要网络）。

验证内容：
1. collection 的显式 schema 与 HNSW + INVERTED 索引能正常建出来；
2. upsert（主键覆盖 = 同名文档重新导入是覆盖不是新增）；
3. 标量过滤（kb_id）在检索中生效，过滤发生在打分之前；
4. 按 document_id 删除清尾巴；
5. 持久化：重建客户端后数据还在（Lite 落盘）。

注意：这里直接注入随机向量，绕开 embedding 调用，
所以账号欠费也能跑——机制和「向量是否真实」是两件事，后者由 ingest 验证。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.console import setup_console  # noqa: E402
from app.vectorstore import MilvusStore, _normalize  # noqa: E402

setup_console()


def fake_vector() -> list[float]:
    return _normalize([random.random() for _ in range(config.EMBED_DIM)])


def make_rows(document_id: str, kb_id: str, n: int, offset: int = 0) -> list[dict]:
    return [
        {
            "chunk_id": f"{document_id}:{offset + i}",
            "vector": fake_vector(),
            "content": f"{document_id} 的第 {offset + i} 块，属于 {kb_id}",
            "kb_id": kb_id,
            "source": f"{document_id}.md",
            "document_id": document_id,
            "chunk_index": offset + i,
            "strategy": "fixed",
        }
        for i in range(n)
    ]


def main() -> None:
    print(f"① 使用 {config.MILVUS_URI}")
    store = MilvusStore()
    store.reset()

    # 写入 docA（kb_a，3 块）与 docB（kb_b，2 块）
    store._insert_rows(make_rows("docA", "kb_a", 3) + make_rows("docB", "kb_b", 2))
    print("② 写入 5 块（docA×3 + docB×2）")

    # 覆盖语义：docA 重新导入只有 2 块（第二块内容变了）。
    # 关键：必须先 delete_document 再 upsert——否则旧的第 2 块会残留成「脏尾巴」
    # （这正是 rag.index_document 里「先删后写」的原因，本测试镜像了该流程）。
    store.delete_document("docA")
    new_a = make_rows("docA", "kb_a", 2)
    new_a[1]["content"] = "docA 的第 1 块（改版后）"
    store._insert_rows(new_a)
    print("③ docA 重新导入为 2 块（模拟改版后变短）")

    # 检索：kb_a 过滤应只返回 kb_a 的两块
    query_vec = fake_vector()
    hits = store.client.search(
        collection_name=config.MILVUS_COLLECTION,
        data=[query_vec],
        limit=10,
        filter=store._expr({"kb_id": "kb_a"}),
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        output_fields=["chunk_id", "kb_id", "content"],
    )[0]
    got = [(h["entity"]["kb_id"], h["entity"]["content"]) for h in hits]
    print(f"④ kb_id 过滤命中 {len(got)} 条（预期 2 条且全部 kb_a）：{got}")

    # 清尾巴：删除 docA 的残留（旧的第 2 块）
    store.delete_document("docA")
    left = store.get_all_chunks()
    print(f"⑤ 删除 docA 后剩余 {len(left)} 块（预期只剩 docB 的 2 块）")

    # 持久化：新建客户端实例，数据仍在
    reopened = MilvusStore()
    print(f"⑥ 重建客户端后仍有 {len(reopened.get_all_chunks())} 块（预期 2）")
    reopened.reset()
    print("⑦ 清理测试数据完成")


if __name__ == "__main__":
    main()
