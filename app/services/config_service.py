"""设备配置的读、校验、写与留痕（AI 助手写操作的唯一入口）。

安全设计（面试重点，对应「写操作需人工确认 + 不注入危险命令」的需求）：
1. **参数白名单**：只允许改登记的少数参数（波特率、上报地址、采样周期），
   白名单外的参数一律拒绝——模型说「帮我把配置全改一遍」时，这是第一道闸；
2. **值域校验**：波特率只能是枚举值；上报地址必须匹配 URL 正则；采样周期只能是枚举值。
   写操作经过两层校验（白名单 + 值域），SQL 里用参数化占位符，
   从「改什么」到「写什么」都不给注入机会；
3. **先读旧值再写**：确认信息里必须展示「原值 → 新值」，用户才知道在确认什么；
4. **写后留痕**：op_log 记录动作、新旧值、操作人、来源（AI 助手）、时间，
   审计可追溯——这是监管场景的硬要求。
"""

from __future__ import annotations

import re
from typing import Any

from .. import database
from ..timeutil import now_str

# 参数白名单：{参数名: (中文名, 值域说明, 校验器)}
# 校验器返回 (合法值, None) 或 (None, 错误说明)
_PORT_RATE = {"4800", "9600", "19200", "38400"}
_INTERVAL = {"1", "5", "15", "30", "60"}
_ADDR_RE = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?(/[\w./-]*)?$")

PARAM_SPECS: dict[str, dict[str, Any]] = {
    "baud_rate": {
        "label": "串口波特率",
        "choices": sorted(_PORT_RATE),
        "check": lambda v: (str(v), None) if str(v) in _PORT_RATE else (None, f"波特率只允许 {sorted(_PORT_RATE)}"),
    },
    "report_addr": {
        "label": "平台上报地址",
        "choices": [],
        "check": lambda v: (
            (str(v).strip(), None)
            if _ADDR_RE.match(str(v).strip())
            else (None, "上报地址必须是 http(s)://主机[:端口][/路径] 形式"),
        ),
    },
    "sample_interval": {
        "label": "采样周期（分钟）",
        "choices": sorted(_INTERVAL),
        "check": lambda v: (str(v), None) if str(v) in _INTERVAL else (None, f"采样周期只允许 {sorted(_INTERVAL)} 分钟"),
    },
}


def get_config(device_id: int, param: str) -> str | None:
    """读参数当前值；未配置过返回 None。"""
    rows = database.query(
        "SELECT value FROM device_config WHERE device_id = ? AND param = ?",
        (device_id, param),
    )
    return rows[0]["value"] if rows else None


def build_proposal(device_id: int, param: str, new_value: str) -> dict[str, Any]:
    """组装「确认书」：把改动讲清楚，供用户确认。校验不通过直接抛 ValueError。"""
    spec = PARAM_SPECS.get(param)
    if spec is None:
        raise ValueError(f"参数 {param} 不在可修改白名单内（可改：{', '.join(PARAM_SPECS)}）")
    clean, err = spec["check"](new_value)
    if err:
        raise ValueError(err)
    old_value = get_config(device_id, param)
    return {
        "ok": True,
        "device_id": device_id,
        "param": param,
        "param_label": spec["label"],
        "old_value": old_value,
        "new_value": str(clean),
        "no_change": old_value is not None and old_value == str(clean),
    }


def apply_change(device_id: int, param: str, new_value: str, operator: str, source: str) -> dict[str, Any]:
    """执行写入并留痕。必须在用户确认之后调用，本函数不做确认。"""
    proposal = build_proposal(device_id, param, new_value)
    if proposal["no_change"]:
        return {**proposal, "applied": False, "message": "新值与当前值相同，未执行写入"}

    exists = database.query(
        "SELECT 1 AS x FROM device_config WHERE device_id = ? AND param = ?",
        (device_id, param),
    ) != []
    if exists:
        database.execute(
            "UPDATE device_config SET value = ?, updated_at = ? WHERE device_id = ? AND param = ?",
            (str(proposal["new_value"]), now_str(), device_id, param),
        )
    else:
        database.execute(
            "INSERT INTO device_config (device_id, param, value, updated_at) VALUES (?, ?, ?, ?)",
            (device_id, param, str(proposal["new_value"]), now_str()),
        )
    database.execute(
        "INSERT INTO op_log (device_id, action, detail, operator, source, ts) VALUES (?, ?, ?, ?, ?, ?)",
        (
            device_id,
            f"config.{param}",
            f"{proposal['old_value']} -> {proposal['new_value']}",
            operator,
            source,
            now_str(),
        ),
    )
    return {**proposal, "applied": True, "message": "修改已生效并写入操作日志"}
