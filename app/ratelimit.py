"""登录限流：按「账号 + 客户端 IP」滑动窗口计数，防暴力破解。

实现选型：单进程内存滑动窗口。多副本部署时必须换 Redis 共享计数
（否则每个副本各自计数，限流形同虚设），与 pending/checkpointer 同一套思路。
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from . import config


class LoginRateLimiter:
    """滑动窗口限流器：窗口内失败次数达到上限即拒绝，成功登录清空计数。"""

    def __init__(self, max_attempts: int = config.LOGIN_RATE_MAX, window_seconds: int = config.LOGIN_RATE_WINDOW):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def allow(self, key: tuple[str, str]) -> bool:
        """记录一次失败并返回是否仍被允许；达到上限返回 False。"""
        now = time.time()
        q = self._events[key]
        while q and q[0] <= now - self.window_seconds:
            q.popleft()
        if len(q) >= self.max_attempts:
            return False
        q.append(now)
        return True

    def clear(self, key: tuple[str, str]) -> None:
        """登录成功后清空计数。"""
        self._events.pop(key, None)


# 进程内单例：auth 模块直接使用
login_limiter = LoginRateLimiter()
