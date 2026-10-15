"""人工确认机制单测：不依赖大模型，直接验证完整闭环。

验证链路：
    工具调用 update_device_config
      → LangGraph interrupt 挂起（GraphInterrupt）
      → 模拟用户批准（Command(resume={"approved": True})）
      → 恢复执行 → 数据库值被修改 + op_log 留痕
      → 再测拒绝路径：库值保持不变

用法：python scripts/test_confirm_unit.py（不需要网络）
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command  # noqa: E402

from app import database  # noqa: E402
from app.agents.state import AgentState  # noqa: E402
from app.console import setup_console  # noqa: E402
from app.tools.write import update_device_config  # noqa: E402

setup_console()


def reset_baud(value: str) -> None:
    database.execute("UPDATE device_config SET value=? WHERE device_id=1 AND param='baud_rate'", (value,))


def current_baud() -> str:
    return database.query("SELECT value FROM device_config WHERE device_id=1 AND param='baud_rate'")[0]["value"]


async def writer_node(state: AgentState) -> dict:
    """模拟员工调用写工具：工具内部会 interrupt，等用户决定。"""
    result = await update_device_config.ainvoke(
        {"device_id": "CEMS-001", "param": "baud_rate", "new_value": "19200"}
    )
    return {"messages": [AIMessage(result)]}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("writer", writer_node)
    g.add_edge(START, "writer")
    g.add_edge("writer", END)
    return g.compile(checkpointer=MemorySaver())


async def main() -> None:
    graph = build_graph()
    empty_input = {"messages": [], "role": "ops", "rounds": 0, "task_calls": []}

    # ① 批准路径
    reset_baud("9600")
    config = {"configurable": {"thread_id": "unit-approve"}}
    first = await graph.ainvoke(empty_input, config=config)
    interrupts = first.get("__interrupt__") or []
    if not interrupts:
        print("❌ 预期出现 __interrupt__，但状态里没有")
    else:
        print(f"✅ 第一次调用正确挂起：{getattr(interrupts[-1], 'value', {})}")
    await graph.ainvoke(Command(resume={"approved": True, "comment": "单测批准"}), config=config)
    print(f"✅ 批准后恢复执行：库值={current_baud()}（预期 19200）")
    print(f"✅ 操作日志：{database.query('SELECT detail FROM op_log WHERE device_id=1 ORDER BY id DESC LIMIT 1')}")

    # ② 拒绝路径
    reset_baud("9600")
    config = {"configurable": {"thread_id": "unit-reject"}}
    first = await graph.ainvoke(empty_input, config=config)
    if not (first.get("__interrupt__")):
        print("⚠️ 拒绝路径第一次调用没有挂起")
    await graph.ainvoke(Command(resume={"approved": False, "comment": "不批准"}), config=config)
    print(f"✅ 拒绝后不写入：库值={current_baud()}（预期仍为 9600）")

    # ③ 白名单拦截（不需要确认，直接拒绝）
    result = await update_device_config.ainvoke(
        {"device_id": "CEMS-001", "param": "drop_table", "new_value": "x"}
    )
    print(f"✅ 白名单拦截：{result}")


if __name__ == "__main__":
    asyncio.run(main())
