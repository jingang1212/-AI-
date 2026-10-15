"""混合检索：中文分词（jieba）+ BM25 关键词召回 + RRF 融合。

- 中文用 jieba（结巴）分词：索引侧 lcut（精确模式），查询侧 cut_for_search
  （搜索引擎模式，会额外产生长词切分，提高召回）；
- BM25 用 BM25L（对长文档有长度修正）；
- RRF 用 1/(k+rank)，k=60，规避向量分与 BM25 分数量纲不同。

jieba 文档：https://github.com/fxsjy/jieba
"""

from __future__ import annotations

import jieba
from rank_bm25 import BM25L

jieba.setLogLevel(20)  # 关掉 jieba 每次加载词典的日志刷屏

RRF_K = 60

def tokenize(text: str, for_search: bool = False) -> list[str]:
    """中文分词：直接用 jieba（缺少依赖会在导入期报错，不降级）。

    for_search=True 用于查询：cut_for_search 会把「设备离线排查」切成
    设备/离线/排查/设备离线/离线排查 这类组合，提高和文档词的命中率。
    """
    cut = jieba.cut_for_search(text) if for_search else jieba.lcut(text)
    return [t.strip().lower() for t in cut if t.strip()]


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
    scores = index.get_scores(tokenize(query, for_search=True))
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [(i, float(scores[i])) for i in ranked if scores[i] > 0][:limit]


def reciprocal_rank_fusion(*ranked_lists: list[str], k: int = RRF_K) -> list[tuple[str, float]]:
    """RRF 融合：按名次打分，两路都靠前的文档得分更高。"""
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, key in enumerate(dict.fromkeys(lst), start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
