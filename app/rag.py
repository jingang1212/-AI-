"""RAG 流水线：解析 → 分块(4 种策略) → 向量化 → 入库 → 检索。"""

from __future__ import annotations

import hashlib
import re
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
    # md / txt 等按 UTF-8 解码；失败时按方案 §4.1 尝试 GBK 严格解码，
    # 两种都失败才报错——严格解码，绝不静默丢字符。
    try:
        return content.decode("utf-8").strip()
    except UnicodeDecodeError:
        pass
    try:
        return content.decode("gbk").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("文档既不是 UTF-8 也不是 GBK 编码，请转换编码后重新上传") from exc


def chunk(
    text: str,
    strategy: str = "fixed",
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """4 种分块策略，返回 [{"text", "start", "end", "heading_path"}]。

    与方案 §4.2 对齐：每个 chunk 都带原文字符区间 start/end 与标题路径
    heading_path（仅 markdown-heading 策略有值），用于检索结果的引用定位。
    """
    text = text.strip()
    if not text:
        return []

    if strategy == "markdown-heading":
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )
        docs = splitter.split_text(text)
        out: list[dict[str, Any]] = []
        search_from = 0
        for d in docs:
            # 分隔符规范成 "  \n"（Markdown 硬换行）会影响原文回读，统一换成普通换行
            piece = d.page_content.replace("  \n", "\n").strip()
            if not piece:
                continue
            # MarkdownHeaderTextSplitter 会把换行规范成 "  \n"，整段未必是原文子串；
            # 按行定位（行内容不变，仅分隔符可能变化）得到字符区间
            start, end = _locate_by_lines(text, piece, search_from)
            # MarkdownHeaderTextSplitter 会维护父级标题：metadata 里带该块所在的 h1/h2/h3
            path = " / ".join(
                d.metadata.get(f"h{i}") for i in (1, 2, 3) if d.metadata.get(f"h{i}")
            )
            out.append({"text": piece, "start": start, "end": end, "heading_path": path})
            search_from = max(0, end)
        return out

    if strategy == "paragraph":
        return [{"heading_path": "", **u} for u in _merge_units(_split_paragraphs(text), size)]

    if strategy == "sentence":
        return [{"heading_path": "", **u} for u in _merge_units(_split_sentences(text), size)]

    # 默认 fixed：递归字符切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", " ", ""],
    )
    pieces = splitter.split_text(text)
    out = []
    search_from = 0
    for piece in pieces:
        start, end = _locate(text, piece, search_from)
        out.append({"text": piece, "start": start, "end": end, "heading_path": ""})
        search_from = max(0, end - overlap)
    return out


def _locate(text: str, piece: str, start_hint: int = 0) -> tuple[int, int]:
    """在原文中顺序定位片段，返回 (start, end)。找不到先退全文字搜索，仍找不到抛错。"""
    idx = text.find(piece, start_hint)
    if idx < 0:
        idx = text.find(piece)
    if idx < 0:
        raise ValueError("无法在原文中定位分块片段：分块结果与原文不一致")
    return idx, idx + len(piece)


def _locate_by_lines(text: str, piece: str, start_hint: int = 0) -> tuple[int, int]:
    """按行定位（分隔符可能被规范化）：首行精确匹配，后续行顺序扩展。"""
    lines = [ln.strip() for ln in piece.split("\n") if ln.strip()]
    if not lines:
        return start_hint, start_hint
    start = text.find(lines[0], start_hint)
    if start < 0:
        # 重叠切分时下一块可能早于上一块终点，退回全文字搜索
        start = text.find(lines[0])
    if start < 0:
        raise ValueError("无法在原文中定位分块片段：分块结果与原文不一致")
    end = start + len(lines[0])
    cursor = end
    for line in lines[1:]:
        pos = text.find(line, cursor)
        if pos < 0:
            break
        end = pos + len(line)
        cursor = end
    return start, end


def _split_paragraphs(text: str) -> list[tuple[str, int, int]]:
    """按空行分段，返回 (段落文本, start, end) 字符区间。"""
    norm = text.replace("\r\n", "\n")
    units: list[tuple[str, int, int]] = []
    for m in re.finditer(r"(?s)(.*?)(?:\n\s*\n|$)", norm):
        raw = m.group(1)
        piece = raw.strip()
        if not piece:
            continue
        start = norm.find(piece, m.start())
        if start < 0 or start >= m.end():
            start = m.start() + len(raw) - len(raw.lstrip())
        units.append((piece, start, start + len(piece)))
    return units


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """按句子边界（。！？）切分，返回 (句子文本, start, end)。"""
    norm = text.replace("\r\n", "\n")
    units: list[tuple[str, int, int]] = []
    for m in re.finditer(r"[^。！？\n]*[。！？]|[^。！？\n]+", norm):
        piece = m.group(0).strip()
        if piece:
            units.append((piece, m.start(), m.end()))
    return units


def _merge_units(units: list[tuple[str, int, int]], size: int) -> list[dict[str, Any]]:
    """把带区间的单元按 size 聚合成块，装不下就新开一块；区间取首尾。"""
    chunks: list[dict[str, Any]] = []
    parts: list[str] = []
    buf_len = 0
    buf_start = 0
    buf_end = 0
    for piece, start, end in units:
        cand_len = len(piece) if not parts else buf_len + 1 + len(piece)
        if parts and cand_len > size:
            chunks.append({"text": "\n".join(parts), "start": buf_start, "end": buf_end})
            parts, buf_len = [piece], len(piece)
            buf_start, buf_end = start, end
        else:
            if not parts:
                buf_start = start
            parts.append(piece)
            buf_len = cand_len
            buf_end = end
    if parts:
        chunks.append({"text": "\n".join(parts), "start": buf_start, "end": buf_end})
    return chunks


def pick_strategy(filename: str, strategy: str | None = None) -> str:
    """未显式指定策略时按文件类型选择：md 用标题分块（带 heading_path），其余固定长度。"""
    if strategy:
        return strategy
    suffix = PurePosixPath(filename).suffix.lower()
    return "markdown-heading" if suffix == ".md" else "fixed"


def _split_parents(text: str, strategy: str) -> list[dict[str, Any]]:
    """父块（给模型的上下文单元）：md 按一级标题分节，其余按固定长度。"""
    if strategy == "markdown-heading":
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1")],
            strip_headers=False,
        )
        parents: list[dict[str, Any]] = []
        for d in splitter.split_text(text):
            piece = d.page_content.replace("  \n", "\n").strip()
            if not piece:
                continue
            heading = d.metadata.get("h1", "")
            if len(piece) > config.PARENT_CHUNK_SIZE:
                # 超长一级节再按固定长度切，继承同一标题路径
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.PARENT_CHUNK_SIZE,
                    chunk_overlap=config.CHUNK_OVERLAP,
                    separators=["\n\n", "\n", "。", " ", ""],
                )
                parents.extend(
                    {"text": s, "heading_path": heading} for s in sub_splitter.split_text(piece)
                )
            else:
                parents.append({"text": piece, "heading_path": heading})
        return parents
    # fixed / paragraph / sentence：沿用 chunk() 作为父块
    return [
        {"text": c["text"], "heading_path": c["heading_path"]}
        for c in chunk(text, strategy=strategy)
    ]


def split_parent_child(text: str, strategy: str = "fixed") -> list[dict[str, Any]]:
    """父子拆分（small-to-big）：父块提供上下文，子块做索引。

    返回子块列表，每个子块带：
    - parent_id / parent_text：所属父块（检索命中后把父块全文喂给模型）；
    - start / end：子块在原文中的字符区间；
    - heading_path：继承父块标题路径。
    """
    parents = _split_parents(text, strategy)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", " ", ""],
    )
    out: list[dict[str, Any]] = []
    search_from = 0
    for pi, parent in enumerate(parents):
        ptext = parent["text"]
        pstart, pend = _locate_by_lines(text, ptext, search_from)
        search_from = max(0, pend - config.CHILD_CHUNK_OVERLAP)
        children = child_splitter.split_text(ptext) or [ptext]
        for ci, child_text in enumerate(children):
            cstart = ptext.find(child_text)
            abs_start = pstart + cstart if cstart >= 0 else pstart
            out.append(
                {
                    "parent_id": f"{pi}",
                    "parent_text": ptext,
                    "text": child_text,
                    "start": abs_start,
                    "end": abs_start + len(child_text),
                    "heading_path": parent["heading_path"],
                }
            )
    return out


def index_document(filename: str, content: bytes, strategy: str = "fixed", kb_id: str = "kb_general") -> dict:
    """上传一个文档并入库。doc_id 用文件名 hash，同名覆盖。

    kb_id 是「标量字段」，检索时可以按它过滤（例如把法规库和运维手册库分开）。
    现在只是写进去，等切到 Milvus 就能直接用 `filter='kb_id == "..."'` 生效——
    这类字段必须入库时就设计好，事后补要全量重灌。
    """
    text = parse(filename, content)
    units = split_parent_child(text, strategy=strategy)
    if not units:
        raise ValueError("文档没有可索引文本")
    doc_id = "doc_" + hashlib.md5(filename.encode()).hexdigest()[:12]
    # 覆盖语义：同名文档的旧 chunk 先清掉，避免改版后残留旧片段
    # （新版 chunk 数变少时，旧的尾巴不清会继续被检索到）
    vectorstore.delete_document(doc_id)
    vectorstore.add_chunks(
        [
            {
                "text": u["text"],
                "metadata": {
                    "document_id": doc_id,
                    "source": filename,
                    "chunk_index": i,
                    "parent_id": f"{doc_id}:{u['parent_id']}",
                    "parent_text": u["parent_text"],
                    "strategy": strategy,
                    "kb_id": kb_id,
                    # 租户隔离预留字段：目前知识库全局共享，统一写 default；
                    # 多租户场景下按租户写入，检索时用 tenant_id 过滤（方案 §3.2）。
                    "tenant_id": "default",
                    "heading_path": u.get("heading_path", ""),
                    "start": u.get("start", 0),
                    "end": u.get("end", 0),
                },
            }
            for i, u in enumerate(units)
        ]
    )
    return {"document_id": doc_id, "chunk_count": len(units)}


def search(query: str, top_k: int = config.DEFAULT_TOP_K, kb_id: str | None = None) -> list[dict]:
    """混合检索：向量召回 + BM25 召回 → RRF 融合 → rerank 精排 → top_k。"""
    from . import hybrid

    # 每个通道各召回 top_k*4 条候选：通道内取多一点，给 RRF 和 rerank 留出腾挪空间。
    # 取 4 倍是经验值——取 1 倍时 rerank 无米下锅，取太多则 rerank 变慢且容易被噪声带偏。
    recall_n = top_k * 4
    where = {"kb_id": kb_id} if kb_id else None

    # 1) 向量召回（语义相近：问「限值多少」能命中「排放浓度不得超过…」）——索引在子块上
    vec_hits = vectorstore.search(query, top_k=recall_n, where=where)
    vec_map = {h["key"]: h for h in vec_hits}

    # 2) BM25 关键词召回（精确匹配：设备型号、标准号、`frame timeout` 这类词向量容易漏）——同样在子块上
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
        meta = chunk["metadata"]
        bm25_map[key] = {
            **chunk,
            "parent_id": meta.get("parent_id"),
            "parent_text": meta.get("parent_text"),
            "score": 0.0,
        }
        if len(bm25_keys) >= recall_n:
            break

    # 3) 子块 → 父块归并：同一父块只保留通道内最高分的子块（排序取首次出现名次）
    def parents_of(keyed: dict[str, dict]) -> tuple[list[str], dict[str, dict]]:
        order: list[str] = []
        best: dict[str, dict] = {}
        for key in keyed:
            h = keyed[key]
            pid = h.get("parent_id") or key  # 没有父块信息时按子块自身
            if pid not in best or (h.get("score") or 0) > (best[pid].get("score") or 0):
                if pid not in best:
                    order.append(pid)
                best[pid] = h
        return order, best

    vec_parent_order, vec_parents = parents_of(vec_map)
    bm25_parent_order, bm25_parents = parents_of(bm25_map)

    # 4) RRF 在父块维度融合
    fused = hybrid.reciprocal_rank_fusion(vec_parent_order, bm25_parent_order)
    all_parents = {**bm25_parents, **vec_parents}  # 向量命中信息更全，后写覆盖
    hits = []
    for pid, rrf_score in fused:
        item = all_parents.get(pid)
        if not item:
            continue
        hits.append(
            {
                **item,
                # 喂给模型的是父块全文（更大的上下文），引用定位仍来自命中的子块
                "text": item.get("parent_text") or item.get("text"),
                "rrf": round(rrf_score, 4),
            }
        )
    hits = hits[:recall_n]

    # 5) rerank 精排：优先精排；失败时退回 RRF 顺序，并给出一次可见警告。
    # rerank 是排序增强器，坏了宁可排序稍差也不要拖垮整个问答；
    # 但降级必须可感知（打印一次原因），不能静默。
    if config.RERANK_MODEL and hits:
        from . import rerank as rerank_mod

        try:
            ranked = rerank_mod.rerank(query, [h["text"] for h in hits], top_n=top_k)
            if ranked:
                hits = [{**hits[i], "rerank": round(score, 4)} for i, score in ranked]
        except Exception as exc:  # noqa: BLE001
            _warn_rerank_failure(exc)
    return hits[:top_k]


_RERANK_WARNED = False


def _warn_rerank_failure(exc: Exception) -> None:
    """rerank 降级只提示一次（避免刷屏），检索继续用 RRF 顺序的结果。"""
    global _RERANK_WARNED
    if _RERANK_WARNED:
        return
    _RERANK_WARNED = True
    from .embeddings import humanize_error

    print(f"⚠️ rerank 调用失败，本次及后续将退回 RRF 排序：{humanize_error(exc)}", file=sys.stderr)
