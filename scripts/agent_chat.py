"""命令行体验多智能体助手（不依赖 Web 服务，方便本地调试）。

用法示例：
    python scripts/agent_chat.py "CEMS-003 这台设备离线了，帮我看看什么原因" --role ops
    python scripts/agent_chat.py "3 号排口最近 24 小时二氧化硫趋势怎么样" --role leader
    python scripts/agent_chat.py --session demo1 "继续说，帮我出个报告" --role leader

--session 就是 thread_id：同一个 session 连续提问，模型会记得上文
（因为 LangGraph 用 thread_id 把对话状态存在 checkpointer 里）。
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.graph import arun  # noqa: E402
from app.auth_context import set_current_user  # noqa: E402
from app.console import setup_console  # noqa: E402
from app.tools import DEFAULT_ROLE, ROLE_PROMPTS  # noqa: E402

setup_console()


def print_events(events: list[dict]) -> None:
    """把执行轨迹打印成人看得懂的进度，便于排查「模型到底调了什么」。"""
    for ev in events:
        kind = ev.get("type")
        if kind == "dispatch":
            print(f"   · 主管派活 → {ev.get('staff')}")
        elif kind == "tool.call":
            print(f"     - {ev.get('staff')} 调用工具 {ev.get('name')} 参数={ev.get('args')}")
        elif kind == "tool.result":
            print(f"     - {ev.get('name')} 返回：{(ev.get('output') or '')[:100]}…")
        elif kind == "reference.source":
            print(f"     - 命中资料：{ev.get('source')}（相关度 {ev.get('score')}）")
        elif kind == "file":
            print(f"     - 生成文件：{ev.get('path')}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="多智能体助手命令行入口")
    parser.add_argument("question", help="要问的问题")
    parser.add_argument("--role", default=DEFAULT_ROLE, choices=sorted(ROLE_PROMPTS), help="用户角色")
    parser.add_argument("--session", default="cli-default", help="会话 ID（thread_id），同 ID 连续对话有记忆")
    args = parser.parse_args()

    # 命令行没有登录流程：注入一个「演示用户」，全量数据范围 + 指定角色。
    # all_access 只存在于 CLI/单测上下文，网络接口永远不会设置它。
    set_current_user({"id": 0, "name": "命令行演示", "role": args.role, "all_access": True})

    print(f"[角色 {args.role} | 会话 {args.session}] 你：{args.question}")
    result = await arun(args.question, thread_id=args.session, role=args.role)
    print("\n执行过程：")
    print_events(result["events"])
    print(f"\n助手（共 {result['rounds']} 轮调度）：\n{result['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
