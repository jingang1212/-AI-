"""设备配置写工具：带人工确认的写操作（D5 核心）。

安全模型（对应需求「修改数据库要让用户确认、不能影响其他方面、不能注入危险命令」）：

    模型说「改波特率」 → 白名单 + 值域校验 → LangGraph interrupt 挂起 →
    前端弹确认（原值 → 新值） → 用户批准 → 参数化 SQL 写入 → op_log 留痕

任何一个环节不同意（白名单外的参数 / 值域不合法 / 用户拒绝），都不会有任何写入。
SQL 全部参数化占位符，且只 UPDATE 单表单行，不提供删除/建表能力——
这就是「不能让 AI 拿着数据库权限乱来」的工程化答案。

官方文档：
- interrupt 人工介入：https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
- Command(resume=...) 恢复执行：https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from langgraph.types import interrupt

from ..agents.trace import trace_event
from ..auth_context import get_current_user
from ..scope import ScopeDenied, require_device
from ..services import config_service, device_service


@tool
async def update_device_config(device_id: str, param: str, new_value: str) -> str:
    """修改一台设备的配置参数（如波特率、上报地址、采样周期）。
    这是写操作：工具会先生成修改提案并**暂停等待用户在前端确认**，
    只有用户点击「批准」后才会真正写入数据库，写入后自动留操作日志。

    Args:
        device_id: 设备编号，例如 CEMS-003 或 3。
        param: 参数名，只支持白名单：baud_rate（波特率）、report_addr（上报地址）、
            sample_interval（采样周期分钟）。白名单之外的一律拒绝。
        new_value: 新值，例如 9600。
    """
    db_id, err = device_service.resolve_device_id(device_id)
    if err:
        return err
    user = get_current_user()
    if user is None:
        return "当前会话没有登录用户，写操作被拒绝。"
    if user["role"] != "ops":
        return f"当前账号角色（{user['role']}）没有修改设备配置的权限。"
    try:
        db_id = require_device(user, db_id)
    except ScopeDenied as exc:
        return str(exc)
    try:
        proposal = config_service.build_proposal(db_id, param, new_value)
    except ValueError as exc:
        # 白名单/值域校验失败：直接转成文字告诉模型，模型会转告用户，不会执行
        return f"修改被拒绝：{exc}"

    if proposal["no_change"]:
        return f"{proposal['param_label']}当前就是 {proposal['new_value']}，无需修改。"

    # 核心：LangGraph interrupt。第一次调用会抛出 GraphInterrupt 把整个图挂起，
    # 恢复执行后返回「用户在前端的决定」。在 CLI/无确认通道场景下会被自动拒绝。
    trace_event("confirm.proposal", proposal=proposal)
    # 注意：payload 里不要用 "type" 字段——graph.arun 会用 trace_event("interrupt", **payload)
    # 转发给前端，若 payload 自带 type 会覆盖外层事件类型，导致前端收不到 interrupt 事件。
    # 这里用 "kind" 描述中断子类型（当前只有 confirm 一种）。
    decision: Any = interrupt(
        {
            "kind": "confirm",
            "title": "AI 助手请求修改设备配置",
            "question": (
                f"是否将设备 {device_id}（ID={db_id}）的「{proposal['param_label']}」"
                f"由 {proposal['old_value']} 改为 {proposal['new_value']}？"
            ),
            "proposal": proposal,
        }
    )
    if not isinstance(decision, dict) or not decision.get("approved"):
        return "用户已拒绝本次修改，未执行任何写入。"

    result = config_service.apply_change(
        db_id, param, str(proposal["new_value"]), operator="web-user", source="AI助手"
    )
    trace_event("write.applied", param=param, new_value=proposal["new_value"], device_id=db_id)
    return (
        f"已把设备 {device_id}（ID={db_id}）的「{proposal['param_label']}」"
        f"从 {proposal['old_value']} 修改为 {proposal['new_value']}，"
        "操作已写入审计日志（op_log），修改后需要重启采集进程生效。"
    )
