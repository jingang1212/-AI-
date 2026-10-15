"""WebSocket 对话端点：/ws/chat

用途（技术方案 §6.4）：
- 微信小程序不支持 SSE/EventSource → 走 WebSocket；
- 需要双向交互：用户可随时发 {"type":"abandon"} 打断当前回答。

消息约定：
- 客户端连上后发第一条 JSON：{"type":"chat","message":"...","session_id":"...","role":"ops"}
- 服务端持续推送统一事件（与 SSE 完全相同的事件模型），直到 complete / error；
- 客户端随时可发 {"type":"abandon"} 让服务端取消本次 Agent 任务。
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from ..agents import graph
from ..agents.trace import MAX_QUEUE_SIZE
from .auth import check_ws_token, resolve_role
from . import pending

# 人工确认等待时长（同 SSE）
CONFIRM_TIMEOUT_SECONDS = 300

router = APIRouter()


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    # 连接阶段先鉴权：不合法直接按 4401 关闭，不浪费一次握手后的处理
    if not check_ws_token(websocket):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()

    # 等第一条业务消息（约定为 JSON）
    try:
        raw = await websocket.receive_text()
        first = json.loads(raw)
    except (WebSocketDisconnect, json.JSONDecodeError):
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        return

    message = str(first.get("message") or "")
    if not message.strip() or first.get("type") != "chat":
        await websocket.send_json({"type": "error", "message": "第一条消息必须是 {\"type\":\"chat\",\"message\":\"...\"}"})
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        return

    role = resolve_role(first.get("role"))
    session_id = str(first.get("session_id") or "default")
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    store = pending.get_store()

    async def decide(payload: dict) -> dict:
        # WebSocket 是双向的，但确认回执统一走 REST（与 SSE 共用一条路径，少一套实现）
        return await store.wait_decision(session_id, CONFIRM_TIMEOUT_SECONDS)

    task = asyncio.create_task(graph.arun(message, session_id, role, queue=queue, decide=decide))

    async def sender() -> None:
        """发送协程：独立于接收协程，业务只往 queue 丢，互不阻塞。"""
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event)
                if event["type"] in ("complete", "error"):
                    break
            try:
                await task
            except Exception as exc:  # noqa: BLE001
                await websocket.send_json({"type": "error", "message": str(exc)[:300]})
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass  # 对端断开时 send 会抛异常，正常退出即可

    send_task = asyncio.create_task(sender())
    try:
        # 接收协程：只处理 abandon；收到断连信号就结束整个会话
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw or "{}")
            if data.get("type") == "abandon":
                await websocket.send_json({"type": "error", "message": "已取消"})
                break
    except WebSocketDisconnect:
        pass
    finally:
        # 清理顺序很重要：先取消业务任务（不再产生事件），再取消发送协程，最后关连接
        if not task.done():
            task.cancel()
        send_task.cancel()
        await store.cleanup(session_id)
        try:
            await websocket.close()
        except RuntimeError:
            pass
