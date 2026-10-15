"""RAG 流水线：解析 → 分块(4 种策略) → 向量化 → 入库 → 检索。"""

from __future__ import annotations

import hashlib
import sys
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from . import config, vectorstore
from .embeddings import embed_texts

# ---------------------------------------------------------------------------
# BM25 索引缓存
# 语料版本号没变就直接复用内存索引。语料几十万字时，重建索引要 1~2 秒纯 CPU 时间，
# 而且这段开销和大模型调用是串行的，用户会明显感觉「第一句回答特别慢」。
# 版本号由 vectorstore 在每次写入/删除后自增，所以入库后会自动失效重建。
# ---------------------------------------------------------------------------
_BM25_CACHE: dict[str, Any] = {"version": -1, "chunks": [], "index": None}


def _bm25_index():
    """返回 (全量 chunk 列表, BM25 索引)。两者顺序必须严格一一对应。"""
    from . import hybrid

    current = vectorstore.version()
    if _BM25_CACHE["version"] != current or _BM25_CACHE["index"] is None:
        chunks = vectorstore.get_all_chunks()
        _BM25_CACHE.update(
            version=current,
            chunks=chunks,
            index=hybrid.build_index([c["text"] for c in chunks]),
        )
    return _BM25_CACHE["chunks"], _BM25_CACHE["index"]


def parse(filename: str, content: bytes) -> str:
    """按后缀解析出纯文本；扫描件/空文本直接报错拦截。"""
    suffix = PurePosixPath(filename).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        text = "\n\n".join((p.extract_text() or "") for p in PdfReader(BytesIO(content)).pages).strip()
        if not text:
            raise ValueError("该 PDF 没有可选择文本，可能是扫描件")
        return text
    if suffix == ".docx":
        from docx import Document
        text = "\n".join(p.text for p in Document(BytesIO(content)).paragraphs).strip()
        if not text:
            raise ValueError("该 Word 文档没有可索引文本")
        return text
    # md / txt 等按 UTF-8 解码
    try:
        return content.decode("utf-8").strip()
    except UnicodeDecodeError:
        return content.decode("gbk", errors="ignore").strip()


def chunk(text: str, strategy: str = "fixed", size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """4 种分块策略：fixed / markdown-heading / paragraph / sentence。"""
    text = text.strip()
    if not text:
        return []

    if strategy == "markdown-heading":
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )
        docs = splitter.split_text(text)
        return [d.page_content.strip() for d in docs if d.page_content.strip()]

    if strategy == "paragraph":
        paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
        return _merge(paras, size)

    if strategy == "sentence":
        import re
        sents = [s.strip() for s in re.split(r"(?<=[。！？])", text) if s.strip()]
        return _merge(sents, size)

    # 默认 fixed：递归字符切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", " ", ""],
    )
    return splitter.split_text(text)


def _merge(units: list[str], size: int) -> list[str]:
    """把单元按 size 聚合成块，装不下就新开一块。"""
    chunks, buf = [], ""
    for u in units:
        cand = f"{buf}\n{u}" if buf else u
        if buf and len(cand) > size:
            chunks.append(buf)
            buf = u
        else:
            buf = cand
    if buf:
        chunks.append(buf)
    return chunks


def index_document(filename: str, content: bytes, strategy: str = "fixed", kb_id: str = "kb_general") -> dict:
    """上传一个文档并入库。doc_id 用文件名 hash，同名覆盖。

    kb_id 是「标量字段」，检索时可以按它过滤（例如把法规库和运维手册库分开）。
    现在只是写进去，等切到 Milvus 就能直接用 `filter='kb_id == "..."'` 生效——
    这类字段必须入库时就设计好，事后补要全量重灌。
    """
    text = parse(filename, content)
    chunks = chunk(text, strategy=strategy)
    if not chunks:
        raise ValueError("文档没有可索引文本")
    doc_id = "doc_" + hashlib.md5(filename.encode()).hexdigest()[:12]
    # 覆盖语义：同名文档的旧 chunk 先清掉，避免改版后残留旧片段
    # （新版 chunk 数变少时，旧的尾巴不清会继续被检索到）
    vectorstore.delete_document(doc_id)
    vectorstore.add_chunks(
        [
            {
                "text": c,
                "metadata": {
                    "document_id": doc_id,
                    "source": filename,
                    "chunk_index": i,
                    "strategy": strategy,
                    "kb_id": kb_id,
                },
            }
            for i, c in enumerate(chunks)
        ]
    )
    return {"document_id": doc_id, "chunk_count": len(chunks)}


def search(query: str, top_k: int = config.DEFAULT_TOP_K, kb_id: str | None = None) -> list[dict]:
    """混合检索：向量召回 + BM25 召回 → RRF 融合 → rerank 精排 → top_k。"""
    from . import hybrid

    # 每个通道各召回 top_k*4 条候选：通道内取多一点，给 RRF 和 rerank 留出腾挪空间。
    # 取 4 倍是经验值——取 1 倍时 rerank 无米下锅，取太多则 rerank 变慢且容易被噪声带偏。
    recall_n = top_k * 4
    where = {"kb_id": kb_id} if kb_id else None

    # 1) 向量召回（语义相近：问「限值多少」能命中「排放浓度不得超过…」）
    vec_hits = vectorstore.search(query, top_k=recall_n, where=where)
    vec_map = {h["key"]: h for h in vec_hits}

    # 2) BM25 关键词召回（精确匹配：设备型号、标准号、`frame timeout` 这类词向量容易漏）
    all_chunks, bm25_index = _bm25_index()
    # 需要按 kb_id 过滤时，先多打分再筛（因为索引是全量的，筛掉一部分后数量会变少）
    bm25_limit = recall_n if not kb_id else min(recall_n * 20, len(all_chunks))
    bm25_keys: list[str] = []
    bm25_map: dict[str, dict] = {}
    for idx, _score in hybrid.rank_with_index(query, bm25_index, limit=bm25_limit):
        chunk = all_chunks[idx]  # 索引下标 = 全量语料下标，两个列表必须同源同顺序
        if kb_id and chunk["metadata"].get("kb_id") != kb_id:
            continue
        key = chunk["key"]
        bm25_keys.append(key)
        bm25_map[key] = {**chunk, "score": 0.0}
        if len(bm25_keys) >= recall_n:
            break

    # 3) RRF 融合
    fused = hybrid.reciprocal_rank_fusion(list(vec_map.keys()), bm25_keys)
    hits = []
    for key, rrf_score in fused:
        item = vec_map.get(key) or bm25_map.get(key)
        if item:
            hits.append({**item, "rrf": round(rrf_score, 4)})
    hits = hits[:recall_n]

    # 4) rerank 精排：配置了就用，失败则退回 RRF 顺序（不阻断检索，但会提示一次失败原因）
    if config.RERANK_MODEL and hits:
        try:
            from . import rerank as rerank_mod

            ranked = rerank_mod.rerank(query, [h["text"] for h in hits], top_n=top_k)
            if ranked:
                hits = [{**hits[i], "rerank": round(score, 4)} for i, score in ranked]
        except Exception as exc:  # noqa: BLE001
            _warn_rerank_failure(exc)
    return hits[:top_k]


_RERANK_WARNED = False


def _warn_rerank_failure(exc: Exception) -> None:
    """rerank 失败只提示一次（避免刷屏），检索本身继续用 RRF 顺序的结果。

    之所以不把 rerank 失败直接抛出去：rerank 是「锦上添花」的精排环节，
    云端不可用时让它拖垮整个问答体验不划算；但也不能完全静默，
    否则用户会以为「检索质量就是这样」，排查不到真正的原因。
    """
    global _RERANK_WARNED
    if _RERANK_WARNED:
        return
    _RERANK_WARNED = True
    from .embeddings import humanize_error

    print(f"⚠️ rerank 调用失败，后续将退回 RRF 排序：{humanize_error(exc)}", file=sys.stderr)
