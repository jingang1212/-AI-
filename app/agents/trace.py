"""执行轨迹（trace）：把「Agent 正在干什么」记录成结构化事件。

为什么需要它？
前端要展示「主管正在调度运维员工…」「正在执行 query_data…」这类进度，
而不是让用户盯着一个转圈等 30 秒。轨迹是流式推送（WebSocket / SSE）的数据源。

实现要点：contextvars.ContextVar
- 一个请求 = 一个 asyncio 任务，任务内所有函数共享同一份 ContextVar；
- asyncio.gather 创建子任务时是「浅拷贝上下文」——拷贝的是变量绑定的
  *同一个 list 对象*，所以子任务往里 append，父任务也能看到；
- 因此不需要把回调一层层往下传参，工具函数直接 trace_event(...) 即可。

官方文档：
- contextvars：https://docs.python.org/3/library/contextvars.html
- asyncio 任务与上下文：https://docs.python.org/3/library/asyncio-task.html#task-object
"""

from __future__ import annotations

import contextvars
import time
from typing import Any

# 默认 None 表示「当前不在一次受追踪的请求里」，此时 trace_event 静默丢弃事件，
# 这样脚本模式（python scripts/agent_chat.py）不传 sink 也能跑。
_TRACE: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "agent_trace", default=None
)


def start_trace() -> list[dict[str, Any]]:
    """开启一次追踪，返回可变的轨迹列表（调用方拿它做推送/落库）。"""
    events: list[dict[str, Any]] = []
    _TRACE.set(events)
    return events


def trace_event(event_type: str, **payload: Any) -> None:
    """记录一个事件，例如 trace_event("tool.call", name="query_data")。

    event_type 命名对齐技术方案 §6.1 的统一事件模型：
    content.delta / tool.call / reference.source / image / interrupt / error / complete
    """
    events = _TRACE.get()
    if events is None:
        return
    events.append({"type": event_type, "ts": time.time(), **payload})
