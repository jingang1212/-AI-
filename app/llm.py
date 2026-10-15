"""对话模型工厂：统一从 .env 构造 ChatOpenAI（走 DashScope 的 OpenAI 兼容模式）。

为什么用 ChatOpenAI 而不是 ChatTongyi（百炼原生 SDK）？
1. 兼容模式把百炼的模型包装成标准 OpenAI 协议，业务代码只认 ChatOpenAI；
   以后换模型厂商（OpenAI / vLLM / Ollama / DeepSeek）只改 .env 里的
   BASE_URL + CHAT_MODEL，一行业务代码都不用动。
2. LangChain 里绝大多数示例、中间件、bind_tools 都是围绕 OpenAI 协议写的，
   选兼容模式能直接复用官方文档的写法。

官方文档：
- 百炼 OpenAI 兼容接口：
  https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope
- LangChain ChatOpenAI 参数（base_url / api_key / temperature / max_retries）：
  https://python.langchain.com/docs/integrations/chat/openai/

注意：必须选「支持 Function Calling」的模型（qwen3.8-flash 支持），
否则 bind_tools 会报错或模型压根不返回 tool_calls，多智能体直接废掉。
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from . import config


class MissingApiKeyError(RuntimeError):
    """没配置 DASHSCOPE_API_KEY 时抛出。

    设计取舍：宁可「启动即报错」，也不要跑到一半在调用大模型时才 401——
    后者会以「工具执行失败」的形式被 Agent 吞掉，排查起来非常痛苦。
    """


def get_chat_model(temperature: float = 0.2, **kwargs) -> ChatOpenAI:
    """创建一个对话模型客户端。

    参数说明（都是官方语义，面试常问）：
    - temperature：采样温度。0 = 尽量确定性输出，适合工具调用/数据查询；
      0.7+ 适合文案创作。Agent 场景一般 0~0.3，避免模型乱编工具参数。
    - timeout：单次请求超时（秒）。Agent 链路上有多次调用，必须设，
      否则一次网络抖动能把整个 WebSocket 连接挂死。
    - max_retries：SDK 层自动重试次数（指数退避）。只重试「可重试错误」
      （超时、429、5xx），不重试 400/401 这类参数或鉴权错误。
    - streaming：这里不开启，流式由上层（LangGraph astream_events / SSE）驱动。

    extra_body={"enable_thinking": ...} 是 DashScope 的私有参数，
    LangChain 没有对应的正式字段，用 extra_body 原样透传（官方也支持这么传）。
    见 config.ENABLE_THINKING 的注释：它直接决定 tool_choice="required" 能不能用。
    """
    if not config.DASHSCOPE_API_KEY:
        raise MissingApiKeyError(
            "缺少 DASHSCOPE_API_KEY：请到阿里云百炼控制台 → API-KEY 管理创建 sk- 开头的 Key，"
            "写入项目根目录 .env 文件（参考 .env.example）。"
        )

    return ChatOpenAI(
        model=config.CHAT_MODEL,
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.BASE_URL,
        temperature=temperature,
        timeout=60,
        max_retries=2,
        extra_body={"enable_thinking": config.ENABLE_THINKING},
        **kwargs,
    )
