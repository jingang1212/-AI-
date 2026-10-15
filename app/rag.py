"""RAG 流水线：解析 → 分块(4 种策略) → 向量化 → 入库 → 检索。"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import PurePosixPath

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from . import config, vectorstore
from .embeddings import embed_texts


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


def index_document(filename: str, content: bytes, strategy: str = "fixed") -> dict:
    """上传一个文档并入库。doc_id 用文件名 hash，同名覆盖。"""
    text = parse(filename, content)
    chunks = chunk(text, strategy=strategy)
    if not chunks:
        raise ValueError("文档没有可索引文本")
    doc_id = "doc_" + hashlib.md5(filename.encode()).hexdigest()[:12]
    vectorstore.get_store().delete(where={"document_id": doc_id})
    vectorstore.add_chunks(
        [
            {
                "text": c,
                "metadata": {
                    "document_id": doc_id,
                    "source": filename,
                    "chunk_index": i,
                    "strategy": strategy,
                },
            }
            for i, c in enumerate(chunks)
        ]
    )
    return {"document_id": doc_id, "chunk_count": len(chunks)}


def search(query: str, top_k: int = config.DEFAULT_TOP_K) -> list[dict]:
    """向量检索（D2 再加 BM25+RRF+rerank）。"""
    return vectorstore.search(query, top_k=top_k)
