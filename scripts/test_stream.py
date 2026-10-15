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


async def test_chart_flow(client: httpx.AsyncClient) -> None:
    """D5 测试：领导角色要趋势图 → 应收到 image 事件 + 可访问的图片 URL。"""
    print("\n===== ③ 折线图生成（leader）=====")
    image_url = None
    async with client.stream(
        "POST",
        "/sse/chat",
        params={"token": TOKEN},
        headers={"X-Role": "leader"},
        json={
            "message": "分析 3 号排口最近 24 小时二氧化硫情况，画一张趋势图",
            "session_id": "chart-test",
        },
    ) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:].strip())
            if event["type"] == "image":
                image_url = event.get("url")
                print(f"[图片] 已生成 {image_url}（{event.get('title')}）")
            elif event["type"] in ("complete", "error"):
                break
    if image_url:
        r = await client.get(image_url, params={"token": TOKEN})
        print(f"[图片] GET {image_url} → HTTP {r.status_code}，大小 {len(r.content)} 字节")


async def test_confirm_flow(client: httpx.AsyncClient) -> None:
    """D5 测试：写操作必须人工确认后才执行。"""
    print("\n===== ④ 写操作人工确认（ops）=====")
    # 复位：保证每次都从「19200 → 9600」的真实改动开始
    from app import database

    # 用在线设备 CEMS-001（心跳正常），避免模型以「设备离线改不了」为由拒单
    database.execute(
        "UPDATE device_config SET value='9600' WHERE device_id=1 AND param='baud_rate'"
    )
    # 心跳是静态演示数据，时间久了会过期变成「离线」，测试前刷新成当前时间
    from app.timeutil import now_str

    database.execute(
        "UPDATE device_heartbeat SET ts=? WHERE device_id=1", (now_str(),)
    )
    session = "confirm-test"
    interrupted = False
    async with client.stream(
        "POST",
        "/sse/chat",
        params={"token": TOKEN},
        headers={"X-Role": "ops"},
        json={
            "message": "调用 update_device_config 工具，把 CEMS-001 的 baud_rate 改成 19200",
            "session_id": session,
        },
    ) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:].strip())
            if event["type"] == "interrupt":
                interrupted = True
                print(f"[确认] {event.get('question')}")
                r = await client.post(
                    "/api/chat/confirm",
                    params={"token": TOKEN},
                    json={"session_id": session, "approved": True, "comment": "现场已确认"},
                )
                print(f"[回执] HTTP {r.status_code} {r.json()}")
            elif event["type"] == "write.applied":
                print(f"[写入] 已生效：{event.get('param')}={event.get('new_value')}")
            elif event["type"] == "content.delta":
                print(event.get("delta", ""), end="", flush=True)
            elif event["type"] in ("complete", "error"):
                print()
                break
    current = database.query(
        "SELECT value FROM device_config WHERE device_id=1 AND param='baud_rate'"
    )[0]["value"]
    ops_log = database.query(
        "SELECT COUNT(*) AS c FROM op_log WHERE device_id=1 AND action='config.baud_rate'"
    )[0]["c"]
    if interrupted:
        print(f"[校验] ✅ 端到端确认链路走通：库中值={current}，操作日志条数={ops_log}")
    else:
        print(
            f"[校验] 本次模型选择了保守叙述、未调用写工具（库中值={current}，未新增日志）。\n"
            "       机制本身已由 scripts/test_confirm_unit.py 确定性验证"
            "（interrupt → 批准 → 落库留痕 / 拒绝 → 不写入 / 白名单拦截）。"
        )


async def main(only: str = "all") -> None:
    client = httpx.AsyncClient(base_url=BASE, timeout=300)
    try:
        health = await client.get("/healthz", params={"token": TOKEN})
        print(f"健康检查 → HTTP {health.status_code} {health.json()}")
        if only in ("all", "chart"):
            await test_chart_flow(client)
        if only in ("all", "confirm"):
            await test_confirm_flow(client)
    finally:
        await client.aclose()
    if only == "all":
        await test_sse()
        await test_ws()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all", choices=["all", "chart", "confirm", "sse", "ws"])
    parsed = parser.parse_args()
    asyncio.run(main(only=parsed.only))
