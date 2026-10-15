"""当前登录用户上下文（ContextVar）。

为什么用 ContextVar 而不是层层传参：
Agent 工具是被 LangChain/LangGraph 动态调用的，函数签名只能有业务参数
（device_id、station_id…），没法塞一个 user 参数。
ContextVar 在请求开始时 set，之后同一条 asyncio 任务树里的所有子任务
都能读到「当前是谁在操作」——和 trace 是同一个套路，已验证可行。
"""

from __future__ import annotations

import contextvars
from typing import Any

_CURRENT_USER: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "current_user", default=None
)


def set_current_user(user: dict[str, Any] | None) -> None:
    _CURRENT_USER.set(user)


def get_current_user() -> dict[str, Any] | None:
    return _CURRENT_USER.get()
