"""时间工具：全项目统一时区口径。

为什么要单独立一个文件？（真实踩坑，面试可讲）
一开始 seed 用 `datetime.now(timezone.utc)` 写入监测数据，而设备实际是在**北京时间**上报的。
后果非常隐蔽：
- 报表里的「08:00 数据」其实是北京时间 16:00 的；
- 用户查「最近 24 小时」，实际窗口和现场感知错开 8 小时，
  现场明明有数据，平台上显示「无数据」，运维白跑一趟；
- 跨零点统计日/小时均值时，超标判定直接算错时段。

正确做法：**库内统一存本地时间字符串（不带时区后缀），全项目用同一个时区生成与解析**，
跨境/多区域场景再改用 UTC 存储 + 展示层转换。

说明：`zoneinfo` 是 Python 3.9+ 标准库，Windows 上需要 `tzdata` 包提供时区数据库，
否则会抛 ZoneInfoNotFoundError——所以这里做了降级兜底。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from . import config

try:
    from zoneinfo import ZoneInfo

    TZ = ZoneInfo(config.TZ_NAME)
except Exception:  # noqa: BLE001 —— 缺 tzdata 时退回固定 +08:00，不影响国内项目使用
    TZ = timezone(timedelta(hours=8))

# 库内时间格式：秒级、不带时区后缀，方便 MySQL/SQLite 直接比较大小
TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def now() -> datetime:
    """当前本地时间（带时区信息）。"""
    return datetime.now(TZ)


def now_str() -> str:
    """当前本地时间的字符串形式。"""
    return now().strftime(TS_FORMAT)


def hours_ago_str(hours: int) -> str:
    """N 小时之前的本地时间字符串，用于 SQL 的 `ts >= ?` 条件。"""
    return (now() - timedelta(hours=hours)).strftime(TS_FORMAT)


def parse(ts: str) -> datetime:
    """把库里的时间字符串解析成带时区的 datetime。

    历史数据可能带 'Z' 或 '+00:00' 后缀（早期版本用 UTC 写入过），
    这里做兼容：带时区的按其自身时区解析，不带的按本地时区解释。
    """
    text = str(ts).strip().replace("T", " ")
    if text.endswith("Z"):
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=TZ)
