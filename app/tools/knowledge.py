"""知识库检索工具：把 RAG 混合检索封装成 LLM 可调用的标准工具。

这是「用 LangChain 封装工具」这条简历点最直接的落点：
模型看到的只有 name / description / args_schema 三样东西，
它靠 description 判断「什么时候该调用我」，靠 args_schema 生成合法参数。

官方文档：
- @tool 装饰器：https://python.langchain.com/docs/how_to/custom_tools/
- 工具描述与参数解析（Google 风格 docstring）：
  https://python.langchain.com/docs/how_to/custom_tools/#structuredtool-docstring

工程细节（面试可讲）：
- docstring 里 `Args:` 段落会被解析成每个参数的描述，模型据此填参数；
- 返回值必须是字符串（或 JSON 字符串）。返回太长会挤爆上下文，
  所以这里对每个片段做截断，并只给「文件名 + 相关度 + 文本」三要素。
"""

from __future__ import annotations

from langchain_core.tools import tool

from .. import config, rag
from ..agents.trace import trace_event

# 单条片段回给模型的字符上限。太小会丢关键结论，太大浪费 token，
# 400 字左右是「一段标准/一条 SOP」的常见长度。
SNIPPET_CHARS = 400


@tool
def knowledge_search(query: str, top_k: int = 5) -> str:
    """在企业知识库中检索资料。适用于：法规政策、排放标准、设备手册、运维 SOP 等
    文档类问题，例如「二氧化硫排放限值是多少」「设备离线怎么排查」「安装步骤有哪些」。
    返回按相关度排序的文档片段，每条都带来源文件名。

    Args:
        query: 检索用的自然语言问题或关键词，越具体越好，例如
            「烟气 CEMS 二氧化硫小时均值排放限值」。
        top_k: 返回条数，默认 5。问「具体数值」时可调小（3）提高精度，
            问「流程步骤」时可调大（8）覆盖更多片段。
    """
    hits = rag.search(query, top_k=top_k)
    if not hits:
        # 明确告诉模型「没查到」，它才知道该转去问用户或换关键词，
        # 而不是凭空编一个答案（幻觉大多来自空检索 + 无提示）。
        return "知识库中没有检索到相关内容，请换用更具体的关键词，或告知用户知识库暂无该资料。"

    lines: list[str] = []
    for i, hit in enumerate(hits, 1):
        # 相关度优先用 rerank 分（若开启），否则退回 RRF 融合分
        score = hit.get("rerank", hit.get("rrf", hit.get("score", 0)))
        text = (hit.get("text") or "").strip().replace("\n", " ")
        lines.append(f"[{i}] 来源：{hit.get('source')}（相关度 {score}）\n{text[:SNIPPET_CHARS]}")
        # 推给前端做「引用来源」展示：用户能点开看原文，也顺手证明检索确实命中
        trace_event("reference.source", source=hit.get("source"), score=score)

    return "\n\n".join(lines)
