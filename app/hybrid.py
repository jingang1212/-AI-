"""混合检索：中文分词 + BM25 关键词召回 + RRF 融合。

参考运维助手项目（agent_py）retrieval/hybrid.py 的实现：
- 中文按「单字 + 相邻二元组」切，英文/日志字段按整词；
- BM25 用 BM25L（对长文档有长度修正）；
- RRF 用 1/(k+rank)，k=60，规避向量分与 BM25 分数量纲不同。
"""

from __future__ import annotations

import re

from rank_bm25 import BM25L

RRF_K = 60

# 英文/数字（含 ._-/ 连接的字段）按整词；中文按连续汉字段
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._:/-][A-Za-z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """中英混合分词：中文单字+二元组，英文整词。"""
    tokens: list[str] = []
    for m in _TOKEN_PATTERN.finditer(text):
        value = m.group(0).lower()
        if "\u3400" <= value[0] <= "\u9fff":
            chars = list(value)
            tokens.extend(chars)
            tokens.extend("".join(chars[i : i + 2]) for i in range(len(chars) - 1))
        else:
            tokens.append(value)
    return tokens


def bm25_rank(query: str, texts: list[str], limit: int = 20) -> list[tuple[int, float]]:
    """BM25L 打分，返回 (index, score)，过滤 0 分（与 query 无词重叠）。"""
    if not texts:
        return []
    return rank_with_index(query, build_index(texts), limit=limit)


def build_index(texts: list[str]) -> BM25L | None:
    """构建 BM25 索引。

    为什么要单独拆出来？（性能坑，面试可讲）
    最初的实现是「每次查询都把全文重新分词 + 重建索引」，
    语料 1 万字时无感，语料几十万字时每次问答要多花 1~2 秒纯 CPU 时间——
    而且这部分开销和大模型调用是串行的，用户感知非常明显。
    拆出来后由调用方按「语料版本号」缓存索引，语料不变就不重建。
    """
    if not texts:
        # rank_bm25 的 BM25L([]) 会直接 ZeroDivisionError（avgdl = num_doc / corpus_size）。
        # 空语料是完全合法的情况（刚初始化、还没导入文档），这里必须挡住，
        # 否则「知识库为空」会以一个看不懂的除零错误暴露给用户。
        return None
    return BM25L([tokenize(t) for t in texts])


def rank_with_index(query: str, index: BM25L | None, limit: int = 20) -> list[tuple[int, float]]:
    """用已有索引打分。返回 (语料下标, 分数)，已按分数降序，0 分（无词重叠）被过滤。"""
    if index is None:
        return []
    scores = index.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [(i, float(scores[i])) for i in ranked if scores[i] > 0][:limit]


def reciprocal_rank_fusion(*ranked_lists: list[str], k: int = RRF_K) -> list[tuple[str, float]]:
    """RRF 融合：按名次打分，两路都靠前的文档得分更高。"""
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, key in enumerate(dict.fromkeys(lst), start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
