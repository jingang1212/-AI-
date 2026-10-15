"""执行轨迹（trace）：把「Agent 正在干什么」记录成结构化事件。

为什么需要它？
前端要展示「主管正在调度运维员工…」「正在执行 query_data…」这类进度，
而不是让用户盯着一个转圈等 30 秒。轨迹是流式推送（WebSocket / SSE）的数据源。

实现要点：contextvars.ContextVar + asyncio.Queue
- 一个请求 = 一个 asyncio 任务，任务内所有函数共享同一份 ContextVar；
- asyncio.gather 创建子任务时是「浅拷贝上下文」——拷贝的是变量绑定的
  *同一个 state 对象*，所以子任务往里 append / put，父任务也能看到；
- 因此不需要把回调一层层往下传参，工具函数直接 trace_event(...) 即可；
- 如果传入了 queue，事件还会被放入队列，由 SSE / WebSocket 端点实时取走推给前端。

官方文档：
- contextvars：https://docs.python.org/3/library/contextvars.html
- asyncio 任务与上下文：https://docs.python.org/3/library/asyncio-task.html#task-object
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from typing import Any

# 默认 None 表示「当前不在一次受追踪的请求里」，此时 trace_event 静默丢弃事件，
# 这样脚本模式（python scripts/agent_chat.py）不传 sink 也能跑。
_TRACE: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "agent_trace", default=None
)

# 事件队列容量：按 token 粒度的事件可能上千条，用有界队列防内存失控；
# 消费端（网络推送）慢于生产端（模型吐字）时，只丢最新事件会丢内容，
# 所以这里选择「满则丢弃」并依赖 10 万上限足够大（实际不会触顶）。
MAX_QUEUE_SIZE = 100_000


def start_trace(queue: asyncio.Queue[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """开启一次追踪，返回可变的轨迹列表（调用方拿它做落库/调试）。

    queue：可选。传入后，每条事件都会实时写入队列，供 SSE/WS 端点流式推给前端。
    """
    state: dict[str, Any] = {"events": [], "queue": queue}
    _TRACE.set(state)
    return state["events"]


def trace_event(event_type: str, **payload: Any) -> None:
    """记录一个事件，例如 trace_event("tool.call", name="query_data")。

    event_type 命名对齐技术方案 §6.1 的统一事件模型：
    content.delta / tool.call / reference.source / image / interrupt / error / complete
    """
    state = _TRACE.get()
    if state is None:
        return
    event: dict[str, Any] = {"type": event_type, "ts": time.time(), **payload}
    state["events"].append(event)
    queue = state.get("queue")
    if queue is not None:
        try:
            # 非阻塞写入：推不动（网络慢）时不让业务执行被拖累，最多丢弃。
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass
