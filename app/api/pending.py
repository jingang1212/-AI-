"""人工确认的挂起登记表：session_id -> Future。

流程（SSE 是单向流，所以「确认回执」走独立 REST 端点）：
1. Agent 里的写工具调用 interrupt → arun 捕获 GraphInterrupt → 发 interrupt 事件；
2. 端点层的 decide 回调在这里登记一个 Future 并 await 它；
3. 前端收到 interrupt 事件后弹确认框，用户点「批准/拒绝」；
4. 前端 POST /api/chat/confirm {session_id, approved}；
5. 该端点 resolve Future → decide 回调返回 → arun 用 Command(resume=...) 恢复执行。

为什么用 Future 而不是全局字典直接存值：
- 等一个 Future 有超时、有取消语义，代码一眼能看懂「谁在等、等什么」；
- 生产环境这里应该换成 Redis（多副本部署时确认请求可能落在另一台机器）。
"""

from __future__ import annotations

import asyncio

_PENDING: dict[str, asyncio.Future] = {}


def register(session_id: str) -> asyncio.Future:
    """登记一个等待确认的会话。重复登记会把旧 Future 判为超时拒绝（防悬挂）。"""
    old = _PENDING.get(session_id)
    if old is not None and not old.done():
        old.set_result({"approved": False, "reason": "被新的确认请求取代"})
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    _PENDING[session_id] = future
    return future


def resolve(session_id: str, decision: dict) -> bool:
    """用用户的决定唤醒等待中的会话。返回是否找到对应会话。"""
    future = _PENDING.pop(session_id, None)
    if future is None or future.done():
        return False
    future.set_result(decision)
    return True


def cleanup(session_id: str) -> None:
    future = _PENDING.pop(session_id, None)
    if future is not None and not future.done():
        future.set_result({"approved": False, "reason": "会话已结束"})
