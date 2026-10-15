"""登录限流单测：不依赖网络。

验证：窗口内失败 N 次后拒绝、成功登录清空后恢复、窗口滑过后重新允许。
用法：python scripts/test_login_rate_limit.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.console import setup_console  # noqa: E402
from app.ratelimit import LoginRateLimiter  # noqa: E402

setup_console()


def main() -> None:
    limiter = LoginRateLimiter(max_attempts=3, window_seconds=60)
    key = ("demo_user", "127.0.0.1")
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is False, "窗口内第 4 次失败应被拒绝"
    print("✅ 窗口内失败 3 次后第 4 次被拒绝")

    limiter.clear(key)
    assert limiter.allow(key) is True, "成功登录清空后应恢复允许"
    print("✅ 成功登录清空计数后恢复")

    short = LoginRateLimiter(max_attempts=3, window_seconds=1)
    assert short.allow(key) and short.allow(key) and short.allow(key)
    time.sleep(1.1)
    assert short.allow(key) is True, "窗口滑过后应重新允许"
    print("✅ 窗口滑过后恢复允许")
    print("全部通过")


if __name__ == "__main__":
    main()
