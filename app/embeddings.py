"""向量化：百炼 text-embedding 模型，走 OpenAI 兼容协议。

为什么不用 langchain_community 的 DashScopeEmbeddings？
- 它需要额外安装 dashscope SDK（多一个依赖、多一层版本兼容问题）；
- 兼容模式已经提供标准的 /compatible-mode/v1/embeddings 接口，
  用 OpenAIEmbeddings 一套写法通吃，以后换厂商只改 .env。
  实测：qwen3.7-text-embedding 返回 1024 维，与方案里的 dim=1024 完全一致。

面试高频考点：**入库和查询必须用同一个 embedding 模型**。
不同模型的向量空间不可比，换模型就等于所有历史向量作废，必须全量重新入库。
**维度也不能变**：Milvus collection 建好 dim=1024 后，传 1536 维会直接报错。
"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from . import config

_emb = None

# 账户类错误的识别关键词 -> 给用户看的人话提示。
# 这类报错必须和「代码写错了」区分开：百炼返回的是 400 + 一长串英文 JSON，
# 新手看到会以为是自己程序崩了，实际只是账户欠费或免费额度用尽。
ACCOUNT_ERROR_HINTS: dict[str, str] = {
    "Arrearage": "阿里云百炼账户欠费或免费额度已用尽 → 请到百炼控制台充值/领取额度后重试",
    "overdue-payment": "阿里云百炼账户欠费 → 请到百炼控制台充值后重试",
    "InvalidApiKey": "API Key 无效 → 请检查 .env 里的 DASHSCOPE_API_KEY 是否复制完整",
}


class EmbeddingError(RuntimeError):
    """向量化失败（异常信息里会尽量给出人话提示）。"""


def humanize_error(exc: Exception) -> str:
    """把厂商的原始报错翻译成可执行的提示，识别不出来就原样返回。"""
    message = str(exc)
    for keyword, hint in ACCOUNT_ERROR_HINTS.items():
        if keyword in message:
            return f"{hint}\n（原始报错：{message[:300]}）"
    return message


def get_embeddings() -> OpenAIEmbeddings:
    """单例 embedding 模型。"""
    global _emb
    if _emb is None:
        if not config.DASHSCOPE_API_KEY:
            raise EmbeddingError(
                "未检测到 DASHSCOPE_API_KEY。请复制 .env.example 为 .env 并填入你的 key。"
            )
        _emb = OpenAIEmbeddings(
            model=config.EMBED_MODEL,
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.BASE_URL,
            # 关键参数，不开会踩坑：LangChain 默认用 tiktoken 预估 token 数，
            # 再按 8191 上限把长文本切开。tiktoken 是 OpenAI 的分词器，
            # 用来估中文/其他厂商模型并不准，容易把一段文档莫名切成多条，
            # 甚至因为分词差异抛异常。关掉后整段交给服务端处理。
            check_embedding_ctx_length=False,
        )
    return _emb


def embed_texts(texts: list[str]) -> list[list[float]]:
    try:
        return get_embeddings().embed_documents(texts)
    except Exception as exc:  # noqa: BLE001 —— 统一包装成人话再抛
        raise EmbeddingError(humanize_error(exc)) from exc


def embed_query(text: str) -> list[float]:
    try:
        return get_embeddings().embed_query(text)
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingError(humanize_error(exc)) from exc
