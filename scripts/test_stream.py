"""双协议端到端测试：SSE + WebSocket。

先启动服务：uvicorn app.api.main:app --port 8000
再跑本脚本：python scripts/test_stream.py

测试内容：
1. 健康检查与鉴权（错误 token 应被拒）；
2. SSE：POST /sse/chat 流式收事件，验证 content.delta 打字机输出与 complete 收尾；
3. WebSocket：/ws/chat 收同样的事件流。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import websockets  # noqa: E402

from app.console import setup_console  # noqa: E402

setup_console()

BASE = "http://127.0.0.1:8000"
TOKEN = "demo-token-please-change"
QUESTION = "CEMS-003 离线了，帮我看看什么原因"
ROLE = "ops"


def show_event(event: dict) -> None:
    kind = event.get("type")
    if kind == "content.delta":
        print(event.get("delta", ""), end="", flush=True)
    elif kind == "dispatch":
        print(f"\n[主管] 派活 → {event.get('staff')}")
    elif kind == "tool.call":
        print(f"[员工] 调工具 {event.get('name')} {event.get('args')}")
    elif kind == "reference.source":
        print(f"[知识] 命中 {event.get('source')}（相关度 {event.get('score')}）")
    elif kind == "complete":
        print("\n[结束] 完整回答已收到")
    elif kind == "error":
        print(f"\n[错误] {event.get('message')}")


async def test_sse() -> None:
    print("===== ① SSE 流式对话 =====")
    # 鉴权失败测试：不带 token 应 401
    async with httpx.AsyncClient(base_url=BASE, timeout=300) as client:
        unauth = await client.post("/sse/chat", json={"message": QUESTION})
        print(f"无 token 请求 → HTTP {unauth.status_code}（预期 401）")

        delta_chars = 0
        async with client.stream(
            "POST",
            "/sse/chat",
            params={"token": TOKEN},
            headers={"X-Role": ROLE},
            json={"message": QUESTION, "session_id": "sse-test"},
        ) as resp:
            print(f"连接 → HTTP {resp.status_code}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:].strip())
                show_event(event)
                if event["type"] == "content.delta":
                    delta_chars += len(event.get("delta") or "")
                if event["type"] in ("complete", "error"):
                    break
        print(f"\nSSE 累计流式输出 {delta_chars} 字")


async def test_ws() -> None:
    print("\n===== ② WebSocket 对话 =====")
    async with websockets.connect(
        f"ws://127.0.0.1:8000/ws/chat?token={TOKEN}", max_size=4 * 1024 * 1024
    ) as ws:
        await ws.send(
            json.dumps({"type": "chat", "message": QUESTION, "session_id": "ws-test", "role": ROLE})
        )
        while True:
            event = json.loads(await ws.recv())
            show_event(event)
            if event["type"] in ("complete", "error"):
                break
    print("\nWebSocket 测试完成")


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE, timeout=30) as client:
        health = await client.get("/healthz", params={"token": TOKEN})
        print(f"健康检查 → HTTP {health.status_code} {health.json()}")
    await test_sse()
    await test_ws()


if __name__ == "__main__":
    asyncio.run(main())
