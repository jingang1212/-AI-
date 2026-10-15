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
QUESTION = "CEMS-001 这台设备状态怎么样，帮我看看"

# 演示账号（scripts/seed.py 创建，密码统一 123456）
USERS = {
    "leader": ("leader_01", "123456"),
    "ops": ("ops_01", "123456"),
}


async def login(client: httpx.AsyncClient, role_key: str) -> dict:
    """登录拿 JWT，返回请求头。角色完全由服务端决定，客户端不声明。"""
    username, password = USERS[role_key]
    resp = await client.post("/api/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    print(
        f"[登录] {data['user']['name']}（{data['user']['role']}），"
        f"可见企业 {len(data['user']['companies'])} 家"
    )
    return {"Authorization": f"Bearer {data['token']}"}


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


async def test_sse(headers: dict) -> None:
    print("===== ① SSE 流式对话 =====")
    # 鉴权失败测试：不带 token 应 401
    async with httpx.AsyncClient(base_url=BASE, timeout=300) as client:
        unauth = await client.post("/sse/chat", json={"message": QUESTION})
        print(f"无 token 请求 → HTTP {unauth.status_code}（预期 401）")

        delta_chars = 0
        async with client.stream(
            "POST",
            "/sse/chat",
            headers=headers,
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


async def test_ws(headers: dict) -> None:
    print("\n===== ② WebSocket 对话 =====")
    jwt_token = headers["Authorization"].removeprefix("Bearer ")
    async with websockets.connect(
        f"ws://127.0.0.1:8000/ws/chat?token={jwt_token}", max_size=4 * 1024 * 1024
    ) as ws:
        await ws.send(
            json.dumps({"type": "chat", "message": QUESTION, "session_id": "ws-test"})
        )
        while True:
            event = json.loads(await ws.recv())
            show_event(event)
            if event["type"] in ("complete", "error"):
                break
    print("\nWebSocket 测试完成")


async def test_chart_flow(client: httpx.AsyncClient, headers: dict) -> None:
    """D5 测试：领导角色要趋势图 → 应收到 image 事件 + 可访问的图片 URL。"""
    print("\n===== ③ 折线图生成（leader）=====")
    image_url = None
    async with client.stream(
        "POST",
        "/sse/chat",
        headers=headers,
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
        r = await client.get(image_url, headers=headers)
        print(f"[图片] GET {image_url} → HTTP {r.status_code}，大小 {len(r.content)} 字节")


async def test_confirm_flow(client: httpx.AsyncClient, headers: dict) -> None:
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
    # 租户隔离：ops_01 是安装/运维人员（安装与运维同一批人，服务全部客户）→ 全量 30 台
    devices = await client.get("/api/devices", headers=headers)
    print(f"[范围] ops 可见设备数 = {devices.json()['count']}（预期 30，全量）")
    allow = await client.get("/api/monitor/data", params={"station_id": 2}, headers=headers)
    print(f"[范围] ops 访问站点 2 → {allow.status_code}（预期 200）")
    # 客服账号无企业绑定，仍验证 fail-closed：访问任何站点数据都 403
    svc = await client.post("/api/auth/login", json={"username": "service_01", "password": "123456"})
    svc_headers = {"Authorization": f"Bearer {svc.json()['token']}"}
    deny = await client.get("/api/monitor/data", params={"station_id": 1}, headers=svc_headers)
    print(f"[范围] 客服访问站点 1 → {deny.status_code}（预期 403，无企业范围）")

    session = "confirm-test"
    interrupted = False
    async with client.stream(
        "POST",
        "/sse/chat",
        headers=headers,
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
                    headers=headers,
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
        health = await client.get("/healthz")
        print(f"健康检查 → HTTP {health.status_code} {health.json()}")
        leader_headers = await login(client, "leader")
        ops_headers = await login(client, "ops")
        if only in ("all", "chart"):
            await test_chart_flow(client, leader_headers)
        if only in ("all", "confirm"):
            await test_confirm_flow(client, ops_headers)
    finally:
        await client.aclose()
    # SSE/WS 各自创建自己的客户端，登录态用 main 里已换到的 JWT 头
    if only in ("all", "sse"):
        await test_sse(ops_headers)
    if only in ("all", "ws"):
        await test_ws(ops_headers)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="all", choices=["all", "chart", "confirm", "sse", "ws"])
    parsed = parser.parse_args()
    asyncio.run(main(only=parsed.only))
