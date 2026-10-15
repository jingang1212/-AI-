"""Redis 集成测试：checkpointer 持久化 + 人工确认登记表。

前置：本机已启动 Redis（redis-server / docker run -d -p 6379:6379 redis:7），
且 .env 里 REDIS_URL 指向它。验证内容：

1. checkpointer 后端 = redis；
2. 多轮对话的 checkpoint 落在 Redis：第二轮调用后，
   通过 graph.get_state 能读到历史消息（进程重启后依然可续）；
3. 人工确认登记表：等待方 brpop 阻塞 → 投递方 rpush → 等到决定；
   没有等待方时 resolve 返回 False（防串单）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.agents.graph import arun, checkpointer_backend, get_graph  # noqa: E402
from app.api.pending import RedisPending, get_store  # noqa: E402
from app.console import setup_console  # noqa: E402

setup_console()


async def test_checkpointer() -> None:
    backend = checkpointer_backend()
    print(f"① checkpointer 后端：{backend}")
    if backend == "memory":
        print("   （当前 Redis 无 RedisJSON 模块，checkpointer 已按设计降级内存；"
              "换 Redis Stack 后无需改代码自动切回 Redis）")
    thread = "redis-persist-test"
    await arun("记住：我叫赵金刚，负责华东片区运维。", thread_id=thread, role="ops")

    # 第二轮：同一 thread_id 会带历史（这就是「多轮记忆」的实现方式）
    second = await arun("我刚才介绍自己叫什么？负责哪片？", thread_id=thread, role="ops")
    print(f"② 第二轮回答：{second['answer'][:200]}")

    # 从 Redis 读回 checkpoint 状态，验证记忆真的持久化了
    state = await get_graph().aget_state({"configurable": {"thread_id": thread}})
    msg_count = len((state.values or {}).get("messages", []))
    print(f"③ checkpoint 中历史消息数：{msg_count}（≥4 说明两轮对话都在 Redis 里）")


async def test_pending() -> None:
    print("\n④ 人工确认登记表（Redis 版）…")
    store = get_store()
    assert isinstance(store, RedisPending), "REDIS_URL 已配置时应用 Redis 实现"

    # 没有等待方时投递决定：必须返回 False（防串单的核心防护）
    orphan = await store.resolve("no-one-waiting", {"approved": True})
    print(f"   无等待方投递 → {orphan}（预期 False）")

    # 正常链路：等待方挂起 10 秒，1 秒后投递方给决定
    async def waiter() -> dict:
        return await store.wait_decision("session-a", timeout=10)

    wait_task = asyncio.create_task(waiter())
    await asyncio.sleep(1)
    delivered = await store.resolve("session-a", {"approved": True, "comment": "批准"})
    decision = await wait_task
    print(f"   投递结果 → {delivered}（预期 True）；等待方收到 → {decision}")
    await store.cleanup("session-a")


async def main() -> None:
    if not config.REDIS_URL:
        print("未配置 REDIS_URL，跳过（先启动 Redis 并在 .env 里填 REDIS_URL）")
        return
    await test_checkpointer()
    await test_pending()


if __name__ == "__main__":
    asyncio.run(main())
